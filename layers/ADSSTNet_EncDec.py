import torch
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_
from .bimamba import MambaBlock, BiSTS
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from .DiffAttn_EncDec import  MultiheadDiffAttn, get_rope_embeddings, MultiheadDiffAttnConv
from utils.tools import _frontal_occipital_indices



#%% Prior-Guided Graph Learning (PGL) module
def _build_fo_restore_mask(num_channels: int, dataset_name: str | None, preserve: bool) -> torch.Tensor:
    """Frontal–Occipital 跨区位置为 True，与 utils.tools 中先验一致；其余为 False。"""
    mask = torch.zeros(num_channels, num_channels, dtype=torch.bool)
    if not preserve:
        return mask
    fo = _frontal_occipital_indices(dataset_name, num_channels)
    if fo is None:
        return mask
    f_idx, o_idx = fo
    for i in f_idx:
        for j in o_idx:
            mask[i, j] = True
            mask[j, i] = True
    return mask
    
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bfn,bnv->bfv', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.relu(h)


def symmetrize(A):
    return 0.5 * (A + A.transpose(-1, -2))


def remove_diag(A):
    return A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))


def topk_sparsify(A, k: int):
    """
    A: (B,N,N) non-negative
    每行保留 top-k，其余置 0（保持可微：用 hard mask，梯度对被mask元素为0）
    """
    if k is None or k <= 0:
        return A
    B, N, _ = A.shape
    vals, idx = torch.topk(A, k=min(k, N), dim=-1)
    mask = torch.zeros_like(A)
    mask.scatter_(-1, idx, 1.0)
    return A * mask


def row_normalize(A, eps=1e-8):
    s = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / s


class PearsonPriorAdj(nn.Module):
    """
    A = alpha * A0 + (1-alpha) * A_learn
    A_learn 来自可学习基座 + 稳定门控的样本相关微调（可选关闭）
    """
    def __init__(
        self,
        A0: torch.Tensor,      # (N,N), 建议已做 abs/阈值/topk/对角0/归一化（至少非负）
        embedding_dim: int,
        dim_UVA: int = 16,
        learn_alpha: bool = True,
        alpha_init: float = 0.95,
        sample_adapt: bool = True,
        topk_A0: int | None = None,
        symmetric: bool = True,
        temperature: float = 1.0,
        dataset_name: str | None = None,
        preserve_frontal_occipital: bool = False,
    ):
        super().__init__()
        assert A0.dim() == 2 and A0.size(0) == A0.size(1)
        self.N = A0.size(0)
        self.embedding_dim = embedding_dim
        self.dim_UVA = dim_UVA
        self.sample_adapt = sample_adapt
        self.topk_A0 = topk_A0
        self.symmetric = symmetric
        self.temperature = temperature
        self.preserve_frontal_occipital = preserve_frontal_occipital

        # Pearson 先验（buffer，不参与梯度）
        self.register_buffer("A0", A0)
        if self.preserve_frontal_occipital:
            self.register_buffer(
                "fo_restore_mask",
                _build_fo_restore_mask(self.N, dataset_name, preserve_frontal_occipital),
            )

        # 可学习图“基座”：低秩因子（比直接学 NxN 稳定）
        # U: (N,r), V: (N,r)  -> logits = U V^T
        self.U = nn.Parameter(0.01 * torch.randn(self.N, self.dim_UVA))
        self.V = nn.Parameter(0.01 * torch.randn(self.N, self.dim_UVA))

        if self.sample_adapt:
            # ——稳定门控：用 LayerNorm + sigmoid，避免 tanh+relu 的“半边死区”
            # gate 输出 (B,N,1)，控制 sample-adapt 的强度
            self.x_norm = nn.LayerNorm(embedding_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.GELU(),
                nn.Linear(embedding_dim // 2, 1)
            )
            # 初始化 gate bias 让 sigmoid 接近 0（不一上来就疯狂改图）
            nn.init.constant_(self.gate_mlp[-1].bias, -2.0)  # sigmoid(-2)≈0.12

            # 将 x 映射到节点向量空间，用于样本相关的 ΔU/ΔV
            self.x_to_node = nn.Linear(embedding_dim, dim_UVA)

        # alpha：控制 A0 vs A_learn 的比例
        if learn_alpha:
            # 用 logit 参数化 alpha，确保在 (0,1)
            # 让初始 alpha≈alpha_init
            alpha_logit = torch.log(torch.tensor(alpha_init) / (1 - torch.tensor(alpha_init)))
            self.alpha_logit = nn.Parameter(alpha_logit)
        else:
            self.register_buffer("alpha_logit", torch.log(torch.tensor(alpha_init) / (1 - torch.tensor(alpha_init))))

    def forward(self, x):
        """
        x: (B,N,seq_len)
        return:
          A: (B,N,N)
          aux: dict with useful regularizers
        """
        B, N, D = x.shape
        assert N == self.N

        # ------ 1) 学到的“基座图” logits_base: (N,N)
        logits_base = self.U @ self.V.t()  # (N,N)

        # ------ 2) 样本相关微调（稳定门控控制强度）
        if self.sample_adapt:
            # gate: (B,N,1) in (0,1)
            x_n = self.x_norm(x)
            gate = torch.sigmoid(self.gate_mlp(x_n))  # (B,N,1)

            # ΔU/ΔV: (B,N,r)
            delta = self.x_to_node(x_n)               # (B,N,r)
            delta = gate * delta                      # 门控后再更新，避免扰动过大

            # 构造样本相关 logits: (B,N,N)
            # logits = (U+ΔU)(V+ΔV)^T
            Ub = self.U.unsqueeze(0) + delta           # (B,N,r)
            Vb = self.V.unsqueeze(0) + delta           # (B,N,r)  (也可用另一套映射更灵活，这里用同一套更稳)
            logits = torch.matmul(Ub, Vb.transpose(1, 2))  # (B,N,N)
        else:
            gate = None
            logits = logits_base.unsqueeze(0).expand(B, -1, -1)

        # ------ 3) non-neg adjacency（用 softplus 比 relu 更平滑）
        A_learn = F.softplus(logits / self.temperature)

        # 可选：对称化 & 去对角
        if self.symmetric:
            A_learn = symmetrize(A_learn)
        A_learn = remove_diag(A_learn)

        # ------ 4) Pearson 先验融合
        alpha = torch.sigmoid(self.alpha_logit)  # scalar in (0,1)
        A0 = self.A0.unsqueeze(0).expand(B, -1, -1)  # (B,N,N)

        # 确保 A0 同样符合约束（如果 A0 里有负值，这里会出问题）
        # 建议你离线把 A0 处理成非负/对角0
        A = alpha * A0 + (1.0 - alpha) * A_learn

        # 可选：Top-k 稀疏化（通常放在融合后更直观）
        if self.topk_A0 is not None:
            A_before_topk = A
            A = topk_sparsify(A, self.topk_A0)
            # 与离线 A0 一致：稀疏后恢复 Frontal–Occipital 跨区权重（取自稀疏前的融合矩阵）
            if self.preserve_frontal_occipital:
                m = self.fo_restore_mask.to(device=A.device, dtype=torch.bool)
                A = torch.where(m, A_before_topk, A)

        # 行归一化（与原实现 softmax(dim=-1) 类似，但不会强制“总是全连接竞争”）
        A = row_normalize(A)

        aux = {
            "delta": (delta.detach() if self.sample_adapt else None),
            "alpha": alpha.detach(),
            "gate": (gate.detach() if gate is not None else None),
            "A_learn": A_learn,   # 需要正则时可用
            "A0": A0,
        }
        return A, aux


class PGL(nn.Module):
    def __init__(
        self,
        seq_len,
        embedding_dim,
        dim_UVA,
        A0: torch.Tensor,
        topk_A0: int | None = None,
        sample_adapt: bool = False,
        d_model: int | None = None,
        dataset_name: str | None = None,
        preserve_frontal_occipital: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.dim_UVA = dim_UVA
        self.sample_adapt = sample_adapt
        self.d_model = d_model

        if self.embedding_dim != self.seq_len:
            ## 低秩瓶颈：seq_len -> r -> embedding_dim，参数量从 O(L*D) 降为 O(L*r + r*D)
            self.seq_to_embed = nn.Sequential(
                nn.Linear(self.seq_len, self.embedding_dim//4),
                nn.Linear(self.embedding_dim//4, self.embedding_dim)
            )
            self.embed_to_seq = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim//4),
                nn.Linear(self.embedding_dim//4, self.seq_len)
            )

        # ——可替换成你的 GCN（保持接口一致）
        self.gcn = GCN(self.embedding_dim, self.embedding_dim, order=2)

        # ——邻接模块：Pearson 先验 + 稳定门控
        self.adj = PearsonPriorAdj(
            A0=A0,
            embedding_dim=self.embedding_dim,
            dim_UVA=self.dim_UVA,
            learn_alpha=True,
            alpha_init=0.95,        # 初期强依赖 Pearson，后面再学
            sample_adapt=self.sample_adapt,      # 想要样本相关图就 True；若跨被试更稳可先 False
            topk_A0=topk_A0,              # 例如 4~8；不想稀疏就 None
            symmetric=True,
            temperature=1.0,
            dataset_name=dataset_name,
            preserve_frontal_occipital=preserve_frontal_occipital,
        )

    def forward(self, x):
        """
        x: (B, seq_len, N)
        return: x_out (B, seq_len, N), adjacency_matrix, aux
        """
        x_graph = x.permute(0, 2, 1)   # (B, N, seq_len)
        if self.embedding_dim != self.seq_len:
            x_proj = self.seq_to_embed(x_graph)   # (B, N, L_proj)
            if x_proj.size(-1) != self.embedding_dim:
                x_proj = F.interpolate(x_proj, size=self.embedding_dim, mode='linear', align_corners=False)
        else:
            x_proj = x_graph

        A, aux = self.adj(x_proj)        # (B, N, N)
        adjacency_matrix = A

        # 支持列表（与你原版一致）
        support = [A]

        x_in = x_proj.permute(0, 2, 1)   # (B, embedding_dim, N)
        x_out = self.gcn(x_in, support)   # (B, embedding_dim, N)
        x_out = x_out.permute(0, 2, 1)    # (B, N, embedding_dim)

        if self.embedding_dim != self.seq_len:
            x_out = self.embed_to_seq(x_out)   # (B, N, L_out)
            if x_out.size(-1) != self.seq_len:
                x_out = F.interpolate(x_out, size=self.seq_len, mode='linear', align_corners=False)
        x_out = x_out.permute(0, 2, 1)    # (B, seq_len, N)

        return x_out, adjacency_matrix, aux


#%%  Adaptive Spectral Modulation (ASM) module
class ASM_fre(nn.Module):
    def __init__(self, seq_len, adaptive_filter=True, sample_adapt_Fre=False):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.sample_adapt_Fre = sample_adapt_Fre
        F = seq_len // 2 + 1
        # 每个频率独立的 frequency_weight，形状 (F, 2) -> view_as_complex 得 (F)
        self.frequency_weight_global = nn.Parameter(torch.randn(seq_len//2+1, 2, dtype=torch.float32) * 0.02)
        self.frequency_weight_local = nn.Parameter(torch.randn(seq_len//2+1, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.frequency_weight_global, std=.02)
        trunc_normal_(self.frequency_weight_local, std=.02)
        self.threshold_param = nn.Parameter(torch.tensor(0.5)) # * 0.5)
        self.threshold_param._no_weight_decay = True

        if self.sample_adapt_Fre:
            # 按样本自适应的动态门控（作用在频率维度）：gate 形状 (B, F, 1)
            self.gate_mlp = nn.Sequential(
                nn.Linear(F * 2, F),  # 输入：每个频率的 magnitude 均值与标准差（沿通道池化）
                nn.GELU(),
                nn.Linear(F, F)
            )
            nn.init.zeros_(self.gate_mlp[-1].bias)  # 初始 gate≈0.5
        else:
            self.fuse_gate = nn.Parameter(torch.zeros(F, 1))

    def _energy_mask(self, x_fft):
        B, _, _ = x_fft.shape
        epsilon = 1e-6

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        normalized_energy = energy / (median_energy + epsilon)

        # 将 normalized_energy 归一化到 [0, 1]（按 batch 做 min-max）
        flat = normalized_energy.view(B, -1)
        min_e = flat.min(dim=1, keepdim=True)[0]
        max_e = flat.max(dim=1, keepdim=True)[0]
        normalized_energy_01 = (normalized_energy - min_e) / (max_e - min_e + epsilon)

        # 将 threshold_param 映射到 [0, 1]（sigmoid）
        threshold_01 = torch.sigmoid(self.threshold_param)

        # 在 0-1 空间做 mask（straight-through）
        adaptive_mask = ((normalized_energy_01 > threshold_01).float() - threshold_01).detach() + threshold_01
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho') # [B, N//2, C]
        weight_global = torch.view_as_complex(self.frequency_weight_global)
        x_weighted = x_fft * weight_global.unsqueeze(0).unsqueeze(-1)

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self._energy_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_local = torch.view_as_complex(self.frequency_weight_local)
            x_weighted2 = x_masked * weight_local.unsqueeze(0).unsqueeze(-1)

            if self.sample_adapt_Fre:
                # 按样本自适应的动态门控（作用在频率维度）：gate 形状 (B, F, 1)
                mag = torch.abs(x_fft)  # (B, F, C)
                mag_mean = mag.mean(dim=-1)   # (B, F) 沿通道池化
                mag_std = mag.std(dim=-1).clamp(min=1e-6)  # (B, F)
                gate_feat = torch.cat([mag_mean, mag_std], dim=-1)  # (B, 2*F)
                gate = torch.sigmoid(self.gate_mlp(gate_feat))  # (B, F)
                gate = gate.unsqueeze(-1)  # (B, F, 1) 广播到 (B, F, C)
                x_weighted = x_weighted + gate * x_weighted2
            else:
                gate = torch.sigmoid(self.fuse_gate).unsqueeze(0).unsqueeze(-1)
                x_weighted = x_weighted + gate * x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class ASM_chan(nn.Module):
    def __init__(self, seq_len, num_channels, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.num_channels = num_channels
        F = seq_len // 2 + 1
        # 每个通道独立的 frequency_weight，形状 (F, C, 2) -> view_as_complex 得 (F, C)
        self.frequency_weight_global = nn.Parameter(torch.randn(num_channels, 2, dtype=torch.float32) * 0.02)
        self.frequency_weight_local = nn.Parameter(torch.randn(num_channels, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.frequency_weight_global, std=.02)
        trunc_normal_(self.frequency_weight_local, std=.02)
        # 每个通道独立的 threshold_param，形状 (C,)
        self.threshold_param = nn.Parameter(torch.full((num_channels,), 0.5, dtype=torch.float32))
        self.threshold_param._no_weight_decay = True

    def _energy_mask(self, x_fft):
        B, F, C = x_fft.shape
        epsilon = 1e-6

        # 每个通道的能量，不沿通道求和：(B, F, C)
        energy = torch.abs(x_fft).pow(2)

        # 按 batch、通道 做 median（沿频率维）
        median_energy = energy.median(dim=1, keepdim=True)[0]  # (B, 1, C)
        normalized_energy = energy / (median_energy + epsilon)

        # 将 normalized_energy 归一化到 [0, 1]（按 batch、通道做 min-max，沿频率维）
        min_e = normalized_energy.min(dim=1, keepdim=True)[0]  # (B, 1, C)
        max_e = normalized_energy.max(dim=1, keepdim=True)[0]  # (B, 1, C)
        normalized_energy_01 = (normalized_energy - min_e) / (max_e - min_e + epsilon)

        # 每个通道的 threshold 映射到 [0, 1]（sigmoid），形状 (C,) -> (1, 1, C)
        threshold_01 = torch.sigmoid(self.threshold_param).view(1, 1, C)

        # 在 0-1 空间做 mask（straight-through），输出 (B, F, C)
        adaptive_mask = ((normalized_energy_01 > threshold_01).float() - threshold_01).detach() + threshold_01

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension，x_fft: (B, F, C)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        # 每个通道的 global 权重，(F, C, 2) -> (F, C) complex，广播为 (1, F, C)
        weight_global = torch.view_as_complex(self.frequency_weight_global)  # (F, C)
        x_weighted = x_fft * weight_global.unsqueeze(0).unsqueeze(0)

        if self.adaptive_filter:
            freq_mask = self._energy_mask(x_fft)  # (B, F, C)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_local = torch.view_as_complex(self.frequency_weight_local)  # (F, C)
            x_weighted2 = x_masked * weight_local.unsqueeze(0).unsqueeze(0)

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x


class ASM_all(nn.Module):
    def __init__(self, seq_len, num_channels, adaptive_filter=True, sample_adapt_Fre=False):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.num_channels = num_channels
        self.sample_adapt_Fre = sample_adapt_Fre
        F = seq_len // 2 + 1
        # 每个通道独立的 frequency_weight，形状 (F, C, 2) -> view_as_complex 得 (F, C)
        self.frequency_weight_global = nn.Parameter(torch.randn(F, num_channels, 2, dtype=torch.float32) * 0.02)
        self.frequency_weight_local = nn.Parameter(torch.randn(F, num_channels, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.frequency_weight_global, std=.02)
        trunc_normal_(self.frequency_weight_local, std=.02)
        # 每个通道独立的 threshold_param，形状 (C,)
        self.threshold_param = nn.Parameter(torch.full((num_channels,), 0.5, dtype=torch.float32))
        self.threshold_param._no_weight_decay = True

        if self.sample_adapt_Fre:
            # 按样本自适应的动态门控（作用在频率维度）：gate 形状 (B, F, 1)
            self.gate_mlp = nn.Sequential(
                nn.Linear(F * 2, F),  # 输入：每个频率的 magnitude 均值与标准差（沿通道池化）
                nn.GELU(),
                nn.Linear(F, F)
            )
            nn.init.zeros_(self.gate_mlp[-1].bias)  # 初始 gate≈0.5
        else:
            self.fuse_gate = nn.Parameter(torch.zeros(F, 1))

    def _energy_mask(self, x_fft):
        B, F, C = x_fft.shape
        epsilon = 1e-6

        # 每个通道的能量，不沿通道求和：(B, F, C)
        energy = torch.abs(x_fft).pow(2)

        # 按 batch、通道 做 median（沿频率维）
        median_energy = energy.median(dim=1, keepdim=True)[0]  # (B, 1, C)
        normalized_energy = energy / (median_energy + epsilon)

        # 将 normalized_energy 归一化到 [0, 1]（按 batch、通道做 min-max，沿频率维）
        min_e = normalized_energy.min(dim=1, keepdim=True)[0]  # (B, 1, C)
        max_e = normalized_energy.max(dim=1, keepdim=True)[0]  # (B, 1, C)
        normalized_energy_01 = (normalized_energy - min_e) / (max_e - min_e + epsilon)

        # 每个通道的 threshold 映射到 [0, 1]（sigmoid），形状 (C,) -> (1, 1, C)
        threshold_01 = torch.sigmoid(self.threshold_param).view(1, 1, C)

        # 在 0-1 空间做 mask（straight-through），输出 (B, F, C)
        adaptive_mask = ((normalized_energy_01 > threshold_01).float() - threshold_01).detach() + threshold_01

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension，x_fft: (B, F, C)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        # 每个通道的 global 权重，(F, C, 2) -> (F, C) complex，广播为 (1, F, C)
        weight_global = torch.view_as_complex(self.frequency_weight_global)  # (F, C)
        x_weighted = x_fft * weight_global.unsqueeze(0)

        if self.adaptive_filter:
            freq_mask = self._energy_mask(x_fft).to(x.device)  # (B, F, C)
            x_masked = x_fft * freq_mask

            weight_local = torch.view_as_complex(self.frequency_weight_local)  # (F, C)
            x_weighted2 = x_masked * weight_local.unsqueeze(0)

            if self.sample_adapt_Fre:
                # 按样本自适应的动态门控（作用在频率维度）：gate 形状 (B, F, 1)
                mag = torch.abs(x_fft)  # (B, F, C)
                mag_mean = mag.mean(dim=-1)   # (B, F) 沿通道池化
                mag_std = mag.std(dim=-1).clamp(min=1e-6)  # (B, F)
                gate_feat = torch.cat([mag_mean, mag_std], dim=-1)  # (B, 2*F)
                gate = torch.sigmoid(self.gate_mlp(gate_feat))  # (B, F)
                gate = gate.unsqueeze(-1)  # (B, F, 1) 广播到 (B, F, C)
                x_weighted = x_weighted + gate * x_weighted2
            else:
                gate = torch.sigmoid(self.fuse_gate).unsqueeze(0)
                x_weighted = x_weighted + gate * x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x, freq_mask if self.adaptive_filter else None, gate if self.sample_adapt_Fre else None



#%%  Adaptive Differential Spectro-Spatio-Temporal Network (ADSSTNet) Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, seq_len, embedding_dim, d_state=16, d_conv_in=4, d_conv=4, d_conv_out=4, dt_rank=1, expand=2,
        bimamba=True, space=True, se_reduction=16, dim_UVA=16, A0=None, topk_A0=6, sample_adapt_A0=False, sample_adapt_Fre=False,
        num_heads=8, i_layer=0, dataset_name: str | None = None, preserve_frontal_occipital: bool = False,
    ):
        super(EncoderLayer, self).__init__()

        self.normASM= RMSNorm(d_model)
        self.asmModule = ASM_all(
            seq_len,
            d_model,
            sample_adapt_Fre=sample_adapt_Fre
        )

        self.normBiSTS = RMSNorm(d_model)
        self.biSTSModule = BiSTS(
            d_model,
            d_state,
            d_conv_in,
            d_conv,
            d_conv_out,
            dt_rank,
            expand,
            bimamba,
            space,
        )

        self.normPGL = RMSNorm(d_model)
        self.pglModule = PGL(
            seq_len,
            embedding_dim,
            dim_UVA,
            A0,
            topk_A0,
            sample_adapt=sample_adapt_A0,
            d_model=d_model,
            dataset_name=dataset_name,
            preserve_frontal_occipital=preserve_frontal_occipital,
        )

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)

        self.channel_fusion = nn.Conv1d(in_channels=d_model*2, out_channels=d_model, kernel_size=d_conv_out, padding='same', groups=d_model, bias=False) # groups=d_model,


    def forward(self, x):
        # x [B, L, D]
        res = x

        asbX = self.normASM(x.to(dtype=self.normASM.weight.dtype))  # [B, L, D] 
        asbX, freq_mask, gate = self.asmModule(asbX)  # [B, L, D]

        mambaX = self.normBiSTS(asbX.to(dtype=self.normBiSTS.weight.dtype))
        mambaX = self.biSTSModule(mambaX) # [B, L, D]

        gcnX = self.normPGL(asbX.to(dtype=self.normPGL.weight.dtype))
        gcnX, adjacency_matrix, aux = self.pglModule(gcnX) # [B, L, D]

        # 1. 相似性感知 (Similarity Sensing)         # 通过线性变换提取用于对比的特征
        q1 = self.q_proj(mambaX) # (B, L, D)
        k2 = self.k_proj(gcnX) # (B, L, D)
        # 计算特征间的点积相似度，并使用 Sigmoid 归一化为门控权重 [b, t, c] # 这里模拟的是“共模分量”的强度：相似度越高，gate越接近1
        common_gate = torch.sigmoid((q1 * k2) / (x.shape[-1] ** 0.5)) # (B, L, D)
        # 2. 交互式差分抑制 (Cross-Inhibition) # 差分公式：从一个信号中减去与另一个信号重叠的部分
        # D = X - Gate * X_neighbor
        mambaX_ = mambaX - common_gate * gcnX
        gcnX_ = gcnX - common_gate * mambaX_
        # 通道维交错拼接: [mamba0, gcn0, mamba1, gcn1, ...]
        channel_fusion = torch.stack((mambaX_, gcnX_), dim=-1).reshape(x.size(0), x.size(1), -1)
        channel_fusion = self.channel_fusion(channel_fusion.transpose(1, 2)).transpose(1, 2)

        x = channel_fusion + res

        ASM_visualization = {'ASM_before': res, 'freq_mask':freq_mask, 'gate':gate, 'ASM_after': asbX}
        PGL_visualization = {'adjacency_matrix':adjacency_matrix, 'aux':aux}
        RGDI_visualization = {'mambaX':mambaX, 'gcnX':gcnX, 'mambaX_':mambaX_, 'gcnX_':gcnX_}
        return x, ASM_visualization, PGL_visualization, RGDI_visualization



#%%  Adaptive Differential Spectro-Spatio-Temporal Network (ADSSTNet) Encoder
class Encoder(nn.Module):
    def __init__(self, e_layers, d_model, seq_len, embedding_dim,
                 d_state=16, d_conv_in=4, d_conv=4, d_conv_out=4, dt_rank=1, expand=2, 
                 bimamba=True, space=True, se_reduction=16, dim_UVA=16, A0=None, topk_A0=6,
                 sample_adapt_A0=False, sample_adapt_Fre=False, num_heads=8,
                 downsample_min_len: int | None = None,
                 dataset_name: str | None = None, preserve_frontal_occipital: bool = True,
    ):
        super(Encoder, self).__init__()
        # none 或 min_len >= seq_len：不下采样，层数 = e_layers
        # min_len < seq_len：层间 AvgPool(2)，反复折半直到 L <= min_len，层数 = 分辨率段数（忽略 e_layers）
        if downsample_min_len is None or downsample_min_len >= seq_len:
            self.num_layers = e_layers
            self.time_pool = nn.AvgPool1d(2, 2)
            L = seq_len
            self.seq_each_layer = [L]
            while len(self.seq_each_layer) < self.num_layers:
                L = max(1, (L - 2) // 2 + 1)
                self.seq_each_layer.append(L)
            self.multiscale = True
            self.output_seq_len = self.seq_each_layer[-1]
        else:
            self.time_pool = nn.AvgPool1d(2, 2)
            L = seq_len
            self.seq_each_layer = [L]
            while L > downsample_min_len:
                L = max(1, (L - 2) // 2 + 1)
                self.seq_each_layer.append(L)
            self.num_layers = len(self.seq_each_layer)
            self.multiscale = True
            self.output_seq_len = self.seq_each_layer[-1]

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model, self.seq_each_layer[i], embedding_dim,
                d_state, d_conv_in, d_conv, d_conv_out, dt_rank, expand, bimamba, space, se_reduction,
                dim_UVA, A0, topk_A0, sample_adapt_A0, sample_adapt_Fre, num_heads, i + 1,
                dataset_name, preserve_frontal_occipital,
            )
            for i in range(self.num_layers)
        ])

    def forward(self, x):
        if not self.multiscale:
            for i_layer, layer in enumerate(self.encoder_layers):
                x, ASM_visualization, PGL_visualization, RGDI_visualization = layer(x)
                # x = layer(x)
            return x, ASM_visualization, PGL_visualization, RGDI_visualization
        out, ASM_visualization_list, PGL_visualization_list, RGDI_visualization_list = [], [], [], []
        # out = []
        for i_layer, layer in enumerate(self.encoder_layers):
            x, ASM_visualization, PGL_visualization, RGDI_visualization = layer(x)
            # x = layer(x)
            out.append(x)
            ASM_visualization_list.append(ASM_visualization)
            PGL_visualization_list.append(PGL_visualization)
            RGDI_visualization_list.append(RGDI_visualization)
            if i_layer < self.num_layers - 1:
                x = self.time_pool(x.transpose(1, 2)).transpose(1, 2)
        return tuple(out), tuple(ASM_visualization_list), tuple(PGL_visualization_list), tuple(RGDI_visualization_list)
