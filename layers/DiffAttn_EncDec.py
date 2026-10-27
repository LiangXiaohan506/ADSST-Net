import math
import torch
import torch.nn.functional as F
from torch import nn

from .rotary import apply_rotary_emb


def get_rope_embeddings(dim, seq_len, theta=10000.0):
    # apply_rotary_emb 要求 cos/sin 为 (seq_len, rotary_dim/2)，且 rotary_dim <= headdim。
    # 奇数 d_model 时若用 arange(0, dim, 2) 会得到 (dim+1)//2 列，rotary_dim 会变成 dim+1 从而断言失败；
    # 故只旋转前 2 * (dim//2) 维，cos/sin 最后一维为 dim//2。
    half = dim // 2
    if half == 0:
        z = torch.zeros(seq_len, 0)
        return z, z.clone()
    rope_full = half * 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_full, 2).float() / float(rope_full)))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        """
        embed_dim: 输入维度
        depth: 当前层索引
        num_heads: 头数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, self.head_dim * self.num_heads * 2, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * self.num_heads * 2, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * self.num_heads * 2, bias=False)
        self.out_proj = nn.Linear(self.head_dim * self.num_heads * 2, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(self, x, rel_pos, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, 2 * self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), 1)
        v = repeat_kv(v.transpose(1, 2), 1)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn


class MultiheadDiffAttnConv(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, d_conv_in=4, d_conv_out=4):
        """
        embed_dim: 输入维度
        depth: 当前层索引
        num_heads: 头数
        d_conv_in: 输入卷积核大小
        d_conv_out: 输出卷积核大小
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Conv1d(embed_dim, self.head_dim * self.num_heads * 2, kernel_size=d_conv_in, padding='same', groups=embed_dim, bias=False)
        self.k_proj = nn.Conv1d(embed_dim, self.head_dim * self.num_heads * 2, kernel_size=d_conv_in, padding='same', groups=embed_dim, bias=False)
        self.v_proj = nn.Conv1d(embed_dim, self.head_dim * self.num_heads * 2, kernel_size=d_conv_in, padding='same', groups=embed_dim, bias=False)
        self.out_proj = nn.Conv1d(self.head_dim * self.num_heads * 2, embed_dim, kernel_size=d_conv_out, padding='same', groups=embed_dim, bias=False)
        # self.out_proj = nn.Linear(self.head_dim * self.num_heads * 2, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, rel_pos, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x.transpose(1, 2)).transpose(1, 2)
        k = self.k_proj(x.transpose(1, 2)).transpose(1, 2)
        v = self.v_proj(x.transpose(1, 2)).transpose(1, 2)

        nh2 = 2 * self.num_heads
        q = torch.stack([q[..., o::nh2] for o in range(nh2)], dim=-2)
        k = torch.stack([k[..., o::nh2] for o in range(nh2)], dim=-2)
        v = torch.stack([v[..., o::nh2] for o in range(nh2)], dim=-2)
        v = v.reshape(bsz, src_len, self.num_heads, 2 * self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), 1)
        v = repeat_kv(v.transpose(1, 2), 1)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn.transpose(1, 2)).transpose(1, 2)
        return attn