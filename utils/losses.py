"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import pdb
from geomloss import SamplesLoss  # 导入 geomloss



#%% 无监督对比学习
# def hierarchical_contrastive_loss(embed_a, embed_b, alpha=0.8, temporal_unit=0, temp=1.0, subject_id=None, gamma=0.0):
#     """
#     Computes a hierarchical contrastive loss across multiple temporal scales.
#     Combines instance-level, temporal-level and optionally subject-level contrastive losses at each scale.

#     Args:
#         embed_a (torch.Tensor): Embeddings from view a, shape [B, T, C].
#         embed_b (torch.Tensor): Embeddings from view b, shape [B, T, C].
#         alpha (float, optional): Weight for instance contrastive loss. Defaults to 0.8.
#         temporal_unit (int, optional): Minimum depth to include temporal loss. Defaults to 0.
#         temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
#         subject_id (torch.Tensor, optional): Subject IDs for each sample, shape [B]. If None, subject loss is skipped.
#         gamma (float, optional): Weight for subject contrastive loss. Defaults to 0.0.

#     Returns:
#         tuple: (instance_loss, temporal_loss, subject_loss, total_loss), each is a scalar torch.Tensor.
#     """
#     loss_instance = torch.tensor(0., device=embed_a.device)
#     loss_temporal = torch.tensor(0., device=embed_a.device)
#     loss_subject = torch.tensor(0., device=embed_a.device)
#     d = 0

#     use_subject = subject_id is not None and gamma != 0

#     while embed_a.size(1) > 1:
#         # Instance-level contrastive loss
#         if alpha != 0:
#             loss_instance += instance_contrastive_loss_mixup(embed_a, embed_b, temp)
#         # Temporal-level contrastive loss (after temporal_unit depth)
#         if d >= temporal_unit and (1 - alpha) != 0:
#             loss_temporal += temporal_contrastive_loss_mixup(embed_a, embed_b, temp)
#         # Subject-level contrastive loss
#         if use_subject:
#             loss_subject += subject_contrastive_loss_mixup(embed_a, embed_b, subject_id, temp)
#         d += 1

#         # Downsample along temporal dimension for next scale
#         embed_a = F.max_pool1d(embed_a.transpose(1, 2), kernel_size=2).transpose(1, 2)
#         embed_b = F.max_pool1d(embed_b.transpose(1, 2), kernel_size=2).transpose(1, 2)

#     # Handle final single-frame case
#     if embed_a.size(1) == 1:
#         if alpha != 0:
#             loss_instance += instance_contrastive_loss_mixup(embed_a, embed_b, temp)
#         if use_subject:
#             loss_subject += subject_contrastive_loss_mixup(embed_a, embed_b, subject_id, temp)
#         d += 1

#     # Normalize by depth and compute total
#     d_safe = max(d, 1)
#     loss_instance = loss_instance / d_safe
#     loss_temporal = loss_temporal / d_safe
#     loss_subject = loss_subject / d_safe
#     loss_total = alpha * loss_instance + (1 - alpha) * loss_temporal + gamma * loss_subject
#     return loss_temporal, loss_instance, loss_subject, loss_total


# def temporal_contrastive_loss_mixup(embed_a, embed_b, temp=1.0):
#     """
#     Computes a temporal contrastive loss with mixup augmentation.
#     Encourages temporal consistency between views along the time dimension.

#     Args:
#         embed_a (torch.Tensor): Embeddings from view a, shape [B, T, C].
#         embed_b (torch.Tensor): Embeddings from view b, shape [B, T, C].
#         temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.

#     Returns:
#         torch.Tensor: The computed temporal contrastive loss.
#     """
#     B, T = embed_a.size(0), embed_a.size(1)
#     alpha = 0.2
#     beta = 0.2

#     if T == 1:
#         return embed_a.new_tensor(0.)

#     # Mixup augmentation along temporal dimension
#     uni_embed_a = alpha * embed_a + (1 - alpha) * embed_a[:, torch.randperm(embed_a.shape[1]), :].view(embed_a.size())
#     uni_embed_b = beta * embed_b + (1 - beta) * embed_b[:, torch.randperm(embed_a.shape[1]), :].view(embed_b.size())

#     # Concatenate original and augmented embeddings
#     z = torch.cat([embed_a, embed_b, uni_embed_a, uni_embed_b], dim=1)

#     # Compute similarity matrix
#     sim = torch.matmul(z[:, : 2 * T, :], z.transpose(1, 2)) / temp  # B x 2T x 2T
#     logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
#     logits += torch.triu(sim, diagonal=1)[:, :, 1:]

#     # Memory optimization for long sequences
#     if T > 1500:
#         z, sim = z.cpu(), sim.cpu()
#         torch.cuda.empty_cache()

#     # Negative log-softmax loss
#     logits = -F.log_softmax(logits, dim=-1)
#     logits = logits[:, :2 * T, :(2 * T - 1)]

#     # Extract positive pair losses and average
#     t = torch.arange(T, device=embed_a.device)
#     loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
#     return loss


# def instance_contrastive_loss_mixup(embed_a, embed_b, temp=1.0):
#     """
#     Computes a sample-level contrastive loss with mixup augmentation.
#     Encourages consistency between views along the batch dimension.

#     Args:
#         embed_a (torch.Tensor): Embeddings from view a, shape [B, T, C].
#         embed_b (torch.Tensor): Embeddings from view b, shape [B, T, C].
#         temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.

#     Returns:
#         torch.Tensor: The computed sample contrastive loss.
#     """
#     B, T = embed_a.size(0), embed_a.size(1)
#     alpha = 0.2
#     beta = 0.2

#     if B == 1:
#         return embed_a.new_tensor(0.)

#     # Mixup augmentation along batch dimension
#     uni_embed_a = alpha * embed_a + (1 - alpha) * embed_a[torch.randperm(embed_a.shape[0]), :, :].view(embed_a.size())
#     uni_embed_b = beta * embed_b + (1 - beta) * embed_b[torch.randperm(embed_b.shape[0]), :, :].view(embed_b.size())

#     # Concatenate and transpose for sample-level comparison
#     z = torch.cat([embed_a, embed_b, uni_embed_a, uni_embed_b], dim=0)
#     z = z.transpose(0, 1)  # T x 2B x C

#     # Compute similarity matrix
#     sim = torch.matmul(z[:, : 2 * B, :], z.transpose(1, 2)) / temp  # T x 2B x 2B
#     logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
#     logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#     logits = -F.log_softmax(logits, dim=-1)
#     logits = logits[:, :2 * B, :(2 * B - 1)]

#     # Extract positive pair losses and average
#     i = torch.arange(B, device=embed_a.device)
#     loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
#     return loss


# def _mixup_partner_diff_subject(subject_id, B, device):
#     """
#     For each batch index k, pick j with subject_id[j] != subject_id[k] (uniform among candidates).
#     If the batch contains only one subject, fall back to j = k.
#     """
#     sub = subject_id.view(-1).long()
#     partner = torch.arange(B, device=device, dtype=torch.long)
#     for k in range(B):
#         cand = torch.nonzero(sub != sub[k], as_tuple=False).view(-1)
#         if cand.numel() > 0:
#             r = torch.randint(0, cand.numel(), (1,), device=device)
#             partner[k] = cand[r]
#     return partner


# def subject_contrastive_loss_mixup(embed_a, embed_b, subject_id, temp=1.0, eps=1e-8):
#     """
#     Computes a subject-level contrastive loss with mixup augmentation.
#     Mixup partners are drawn from **different** subjects when possible.
#     Positive pairs are only among the first two blocks (embed_a / embed_b): same subject, not self.

#     Args:
#         embed_a (torch.Tensor): Embeddings from view a, shape [B, T, C].
#         embed_b (torch.Tensor): Embeddings from view b, shape [B, T, C].
#         subject_id (torch.Tensor): Subject IDs for each sample, shape [B].
#         temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
#         eps (float, optional): Small constant for numerical stability. Defaults to 1e-8.

#     Returns:
#         torch.Tensor: The computed subject contrastive loss.
#     """
#     B, T, C = embed_a.size(0), embed_a.size(1), embed_a.size(2)
#     alpha = 0.2
#     beta = 0.2

#     if B == 1:
#         return embed_a.new_tensor(0.)

#     device = embed_a.device
#     subject_id = subject_id.to(device).long()

#     # Mixup: interpolate with another sample from a **different** subject (per row)
#     pa = _mixup_partner_diff_subject(subject_id, B, device)
#     pb = _mixup_partner_diff_subject(subject_id, B, device)
#     uni_embed_a = alpha * embed_a + (1 - alpha) * embed_a[pa]
#     uni_embed_b = beta * embed_b + (1 - beta) * embed_b[pb]

#     # Concatenate: [embed_a, embed_b, uni_embed_a, uni_embed_b] -> (4B, T, C)
#     z = torch.cat([embed_a, embed_b, uni_embed_a, uni_embed_b], dim=0)
#     z = z.transpose(0, 1)  # (T, 4B, C)

#     # Compute similarity matrix per timestep: (T, 4B, 4B)
#     sim = torch.matmul(z, z.transpose(1, 2)) / temp

#     logits_mask = torch.ones(4 * B, 4 * B, device=device)
#     logits_mask.fill_diagonal_(0.0)

#     # Positives only in embed_a ∪ embed_b (first 2B rows/cols): same subject, not self.
#     sub2 = torch.cat([subject_id, subject_id], dim=0)  # (2B,)
#     same_sub_2b = (sub2.unsqueeze(0) == sub2.unsqueeze(1)).float()
#     eye_2b = torch.eye(2 * B, device=device, dtype=same_sub_2b.dtype)
#     mask_2b = same_sub_2b * (1.0 - eye_2b)
#     mask = torch.zeros(4 * B, 4 * B, device=device, dtype=mask_2b.dtype)
#     mask[: 2 * B, : 2 * B] = mask_2b
#     mask = mask * logits_mask

#     # SupCon-style loss: -log(mean over positives of softmax)
#     sim = sim - sim.max(dim=2, keepdim=True).values.detach()
#     exp_logits = torch.exp(sim) * logits_mask.unsqueeze(0)
#     log_prob = sim - torch.log(exp_logits.sum(dim=2, keepdim=True) + eps)

#     pos_count = mask.sum(dim=1)
#     denom = pos_count.clamp_min(1.0)
#     mean_log_prob_pos = (mask.unsqueeze(0) * log_prob).sum(dim=2) / denom.unsqueeze(0)

#     valid = pos_count > 0
#     if valid.any():
#         loss = -mean_log_prob_pos[:, valid].mean()
#     else:
#         loss = embed_a.new_tensor(0.)
#     return loss


#%% ===========================================================================================================================
def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(
            divide_no_nan(
                t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)
            )
            * mask
        )


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


#%% Supervised Contrastive Loss (有监督对比损失) Universum样本混合
def _mixup_partner_diff_label(labels: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
    """
    For each batch index i, pick j with labels[j] != labels[i] (uniform over candidates).
    If the batch has only one class, fall back to uniform random j in [0, B).

    Args:
        labels: [B] class indices (caller ensures shape / device consistency).
        B: batch size.
        device: tensor device for sampled indices.

    Returns:
        Tensor of shape [B], partner index per row (same dtype long, on ``device``).
    """
    labels = labels.to(device=device, dtype=torch.long).view(B)
    label_diff_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    perm_indices = []
    for i in range(B):
        diff_idx = label_diff_mask[i].nonzero(as_tuple=False).view(-1)
        if diff_idx.numel() > 0:
            r = torch.randint(0, diff_idx.numel(), (1,), device=device)
            perm_indices.append(diff_idx[r])
        else:
            perm_indices.append(torch.randint(0, B, (1,), device=device))
    return torch.cat(perm_indices)


def _mixup_partners_diff_subject_label(sub: torch.Tensor, labels: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
    """
    For each batch index ``k``, sample one partner index ``partner[k]``.

    Preference order:
        1. Uniform over samples with **different subject** and **different label** from ``k``.
        2. Else uniform over samples with **different subject** only.
        3. Else leave ``partner[k] == k`` (identity; no alternative in-batch).

    Args:
        sub: Subject ids, shape ``[B]`` (caller may rely on ``.to(device, dtype=long)``).
        labels: Class labels, shape ``[B]``.
        B: Batch size.
        device: Device for index tensors and ``torch.randint``.

    Returns:
        ``partner``: Long tensor of shape ``[B]`` on ``device``, mixup index per row.
    """
    sub = sub.to(device=device, dtype=torch.long).view(B)
    labels = labels.to(device=device, dtype=torch.long).view(B)
    diff_label_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    diff_sub_mask = sub.unsqueeze(0) != sub.unsqueeze(1)
    target_mask = diff_label_mask & diff_sub_mask

    partner = torch.arange(B, device=device, dtype=torch.long)

    for k in range(B):
        cand = torch.nonzero(target_mask[k], as_tuple=False).view(-1)
        if cand.numel() == 0:
            cand = torch.nonzero(diff_sub_mask[k], as_tuple=False).view(-1)
        if cand.numel() > 0:
            r = torch.randint(0, cand.numel(), (1,), device=device)
            partner[k] = cand[r]

    return partner


def temporal_supcon_loss_universum(embed_ori, embed_aug, supcon_temp=0.1, supcon_alpha=0.2, temporal_unit=0):
    """
    Computes a temporal contrastive loss with mixup augmentation.
    Encourages temporal consistency between views along the time dimension.

    Args:
        embed_ori (torch.Tensor): Embeddings from view a, shape [B, T, C].
        embed_aug (torch.Tensor): Embeddings from view b, shape [B, T, C].
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
        supcon_alpha (float, optional): Mixup coefficient for Universum. Defaults to 0.2.
        temporal_unit (int, optional): Minimum depth to include temporal loss. Defaults to 0.

    Returns:
        torch.Tensor: The computed temporal contrastive loss.
    """
    B, seq_len = embed_ori.size(0), embed_ori.size(1)

    if seq_len == 1:
        return embed_ori.new_tensor(0.)

    # Mixup augmentation along temporal dimension
    uni_embed_ori = supcon_alpha * embed_ori + (1 - supcon_alpha) * embed_ori[:, torch.randperm(embed_ori.shape[1]), :].view(embed_ori.size())
    uni_embed_aug = supcon_alpha * embed_aug + (1 - supcon_alpha) * embed_aug[:, torch.randperm(embed_aug.shape[1]), :].view(embed_aug.size())

    # Concatenate original and augmented embeddings
    z = torch.cat([embed_ori, embed_aug, uni_embed_ori, uni_embed_aug], dim=1)

    # Compute similarity matrix
    sim = torch.matmul(z[:, : 2 * seq_len, :], z.transpose(1, 2)) / supcon_temp  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    # Memory optimization for long sequences
    if seq_len > 1500:
        z, sim = z.cpu(), sim.cpu()
        torch.cuda.empty_cache()

    # Negative log-softmax loss
    logits = -F.log_softmax(logits, dim=-1)
    logits = logits[:, :2 * seq_len, :(2 * seq_len - 1)]

    # Extract positive pair losses and average
    t = torch.arange(seq_len, device=embed_ori.device)
    loss = (logits[:, t, seq_len + t - 1].mean() + logits[:, seq_len + t, t].mean()) / 2
    return loss


def instance_supcon_loss_universum(embed_ori, embed_aug, labels, supcon_temp=0.1, supcon_alpha=0.2, perm_indices=None):
    """
    Computes a sample-level contrastive loss with mixup augmentation.
    Encourages consistency between views along the batch dimension.
    
    Args:
        embed_ori (torch.Tensor): 视图 a 的嵌入, [B, T, C]
        embed_aug (torch.Tensor): 视图 b 的嵌入, [B, T, C]
        labels (torch.Tensor): 样本标签, [B]
        supcon_temp (float): 温度参数 (有监督模式通常建议设小一点，如 0.07)
        supcon_alpha (float): Mixup 系数
        perm_indices (torch.Tensor, optional): Mixup indices, shape [B]. If None, perm_indices will be computed.
    """
    B, T, C = embed_ori.size(0), embed_ori.size(1), embed_ori.size(2)
    device = embed_ori.device
    dtype = embed_ori.dtype
    labels = labels.to(device=device, dtype=torch.long).view(B)

    if B <= 1:
        return embed_ori.new_tensor(0.)

    if perm_indices is None:
        with torch.no_grad():
            perm_indices = _mixup_partner_diff_label(labels, B, device)

    uni_ori = supcon_alpha * embed_ori + (1 - supcon_alpha) * embed_ori[perm_indices, :, :]
    uni_aug = supcon_alpha * embed_aug + (1 - supcon_alpha) * embed_aug[perm_indices, :, :]

    z_ori = F.normalize(embed_ori, p=2, dim=-1)
    z_aug = F.normalize(embed_aug, p=2, dim=-1)
    z_uni_ori = F.normalize(uni_ori, p=2, dim=-1)
    z_uni_aug = F.normalize(uni_aug, p=2, dim=-1)

    # [4B, T, C] -> [T, 4B, C]，每个 t 上计算 (4B × 4B) 样本间相似度
    features = torch.cat([z_ori, z_aug, z_uni_ori, z_uni_aug], dim=0)
    z = features.transpose(0, 1).contiguous()
    sim_matrix = torch.bmm(z, z.transpose(1, 2)) / supcon_temp  # [T, 4B, 4B]

    labels_view = labels.contiguous().view(-1, 1)
    full_labels = torch.cat([labels_view, labels_view], dim=0)
    mask = (full_labels == full_labels.T).to(dtype=dtype)
    logits_mask = torch.ones_like(mask)
    logits_mask.fill_diagonal_(0)
    mask = mask * logits_mask

    logits_max, _ = torch.max(sim_matrix[:, : 2 * B, :], dim=2, keepdim=True)
    logits = sim_matrix[:, : 2 * B, :] - logits_max.detach()
    exp_logits = torch.exp(logits)

    full_logits_mask = torch.ones((T, 2 * B, 4 * B), device=device, dtype=dtype)
    full_logits_mask[:, :, : 2 * B] = logits_mask.unsqueeze(0).expand(T, -1, -1)

    log_prob = logits - torch.log((exp_logits * full_logits_mask).sum(2, keepdim=True) + 1e-6)

    mean_log_prob_pos = (mask.unsqueeze(0) * log_prob[:, :, : 2 * B]).sum(2) / (
        mask.sum(1).unsqueeze(0) + 1e-6
    )
    loss = -mean_log_prob_pos.mean(dim=1).mean()
    return loss


def subject_supcon_loss_universum(embed_ori, embed_aug, subject_id, labels, supcon_temp=0.1, supcon_alpha=0.2, partner=None):
    """
    Computes a subject-level contrastive loss with mixup augmentation.
    Mixup partners are drawn from **different** subjects when possible.
    Positive pairs are only among the first two blocks (embed_ori / embed_aug): same subject, not self.

    Args:
        embed_ori (torch.Tensor): Embeddings from view a, shape [B, T, C].
        embed_aug (torch.Tensor): Embeddings from view b, shape [B, T, C].
        subject_id (torch.Tensor): Subject IDs for each sample, shape [B].
        labels (torch.Tensor): Labels for each sample, shape [B].
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
        supcon_alpha (float, optional): Mixup coefficient for Universum. Defaults to 0.2.
        partner (torch.Tensor, optional): Mixup indices, shape [B]. If None, partner will be computed.

    Returns:
        torch.Tensor: The computed subject contrastive loss.
    """
    B, T, C = embed_ori.size(0), embed_ori.size(1), embed_ori.size(2)
    device = embed_ori.device
    dtype = embed_ori.dtype
    sub = subject_id.to(device=device, dtype=torch.long).view(B)
    labels = labels.to(device=device, dtype=torch.long).view(B)

    if B <= 1:
        return embed_ori.new_tensor(0.)

    if partner is None:
        with torch.no_grad():
            partner = _mixup_partners_diff_subject_label(sub, labels, B, device)

    uni_embed_ori = supcon_alpha * embed_ori + (1 - supcon_alpha) * embed_ori[partner, :, :]
    uni_embed_aug = supcon_alpha * embed_aug + (1 - supcon_alpha) * embed_aug[partner, :, :]

    z_ori = F.normalize(embed_ori, p=2, dim=-1)
    z_aug = F.normalize(embed_aug, p=2, dim=-1)
    z_uni_ori = F.normalize(uni_embed_ori, p=2, dim=-1)
    z_uni_aug = F.normalize(uni_embed_aug, p=2, dim=-1)

    z = torch.cat([z_ori, z_aug, z_uni_ori, z_uni_aug], dim=0)
    z = z.transpose(0, 1).contiguous()
    sim = torch.bmm(z, z.transpose(1, 2)) / supcon_temp  # [T, 4B, 4B]

    sub2 = torch.cat([sub, sub], dim=0)
    mask_2b = (sub2.unsqueeze(0) == sub2.unsqueeze(1)).to(dtype=dtype)
    mask_2b = mask_2b * (1.0 - torch.eye(2 * B, device=device, dtype=dtype))

    mask = torch.zeros(4 * B, 4 * B, device=device, dtype=dtype)
    mask[: 2 * B, : 2 * B] = mask_2b

    logits_mask = torch.ones(4 * B, 4 * B, device=device, dtype=dtype)
    logits_mask.fill_diagonal_(0.0)
    logits_mask_t = logits_mask.unsqueeze(0).expand(T, -1, -1)

    sim_max, _ = torch.max(sim, dim=2, keepdim=True)
    logits = sim - sim_max.detach()
    exp_logits = torch.exp(logits) * logits_mask_t
    log_prob = logits - torch.log(exp_logits.sum(dim=2, keepdim=True) + 1e-6)

    pos_count = mask.sum(dim=1)
    anchor_valid = pos_count > 0
    anchor_valid_2b = anchor_valid[: 2 * B]

    if not anchor_valid_2b.any():
        return embed_ori.new_tensor(0.)

    numer = (mask.unsqueeze(0) * log_prob).sum(dim=2)
    mean_log_prob_pos = numer[:, : 2 * B] / pos_count[: 2 * B].clamp(min=1.0).unsqueeze(0)
    loss = -mean_log_prob_pos[:, anchor_valid_2b].mean(dim=1).mean()
    return loss


def hierarchical_supcon_loss_universum(
    embed_ori, embed_aug, labels, subject_id,
    lambda_temporal=0.0, lambda_instance=0.0, lambda_subject=0.0,
    temporal_unit=0, supcon_temp=1.0, supcon_alpha=0.2):
    """
    Computes a hierarchical contrastive loss across multiple temporal scales.
    Combines instance-level, temporal-level and optionally subject-level contrastive losses at each scale.

    Args:
        embed_ori (torch.Tensor): Embeddings from view a, shape [B, T, C].
        embed_aug (torch.Tensor): Embeddings from view b, shape [B, T, C].
        labels (torch.Tensor): Labels for each sample, shape [B].
        subject_id (torch.Tensor, optional): Subject IDs for each sample, shape [B]. If None, subject loss is skipped.
        lambda_temporal (float, optional): Weight for temporal contrastive loss. Defaults to 0.0.
        lambda_instance (float, optional): Weight for instance contrastive loss. Defaults to 0.0.
        lambda_subject (float, optional): Weight for subject contrastive loss. Defaults to 0.0.
        temporal_unit (int, optional): Minimum depth to include temporal loss. Defaults to 0.
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
        supcon_alpha (float, optional): Mixup coefficient for Universum. Defaults to 0.2.

    Returns:
        tuple: (instance_loss, temporal_loss, subject_loss, total_loss), each is a scalar torch.Tensor.
    """
    loss_instance = torch.tensor(0., device=embed_ori.device)
    loss_temporal = torch.tensor(0., device=embed_ori.device)
    loss_subject = torch.tensor(0., device=embed_ori.device)
    d_temp, d_inst = 0, 0

    with torch.no_grad():
        perm_indices = _mixup_partner_diff_label(labels, embed_ori.size(0), embed_ori.device)
        partner = _mixup_partners_diff_subject_label(subject_id, labels, embed_ori.size(0), embed_ori.device)

    while embed_ori.size(1) > 1:
        # Temporal-level contrastive loss
        if lambda_temporal != 0 and d_temp >= temporal_unit:
            loss_temporal += temporal_supcon_loss_universum(
                embed_ori, embed_aug, supcon_temp=supcon_temp, supcon_alpha=supcon_alpha, temporal_unit=temporal_unit,
            )
        # Instance-level contrastive loss
        if lambda_instance != 0:
            loss_instance += instance_supcon_loss_universum(
                embed_ori, embed_aug, labels, supcon_temp=supcon_temp, supcon_alpha=supcon_alpha, perm_indices=perm_indices
            )
        # Subject-level contrastive loss
        if lambda_subject != 0:
            loss_subject += subject_supcon_loss_universum(
                embed_ori, embed_aug, subject_id, labels, supcon_temp=supcon_temp, supcon_alpha=supcon_alpha, partner=partner,
            )
        d_temp += 1
        d_inst += 1

        embed_ori = F.max_pool1d(embed_ori.transpose(1, 2), kernel_size=2).transpose(1, 2)
        embed_aug = F.max_pool1d(embed_aug.transpose(1, 2), kernel_size=2).transpose(1, 2)

    # Handle final single-frame case
    if embed_ori.size(1) == 1:
        if lambda_instance != 0:
            loss_instance += instance_supcon_loss_universum(
                embed_ori, embed_aug, labels, supcon_temp=supcon_temp, supcon_alpha=supcon_alpha, perm_indices=perm_indices
            )
        if lambda_subject != 0:
            loss_subject += subject_supcon_loss_universum(
                embed_ori, embed_aug, subject_id, labels, supcon_temp=supcon_temp, supcon_alpha=supcon_alpha, partner=partner,
            )
        d_inst += 1

    # Normalize by depth and compute total
    loss_temporal = loss_temporal / d_temp
    loss_instance = loss_instance / d_inst
    loss_subject = loss_subject / d_inst
    loss_total = lambda_temporal * loss_temporal + lambda_instance * loss_instance + lambda_subject * loss_subject
    return loss_temporal, loss_instance, loss_subject, loss_total


#%% 有监督对比学习 没有Universum样本混合
def temporal_supcon_loss(embed_ori, embed_aug, supcon_temp=0.1):
    """
    Computes a temporal contrastive loss.
    Encourages temporal consistency between views along the time dimension.

    Args:
        embed_ori (torch.Tensor): Embeddings from view a, shape [B, T, C].
        embed_aug (torch.Tensor): Embeddings from view b, shape [B, T, C].
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.

    Returns:
        torch.Tensor: The computed temporal contrastive loss.
    """
    B, seq_len = embed_ori.size(0), embed_ori.size(1)

    if seq_len == 1:
        return embed_ori.new_tensor(0.)

    # L2 normalize on feature dimension
    z_ori = F.normalize(embed_ori, p=2, dim=-1)
    z_aug = F.normalize(embed_aug, p=2, dim=-1)
        
    # Concatenate original and augmented embeddings
    z = torch.cat([z_ori, z_aug], dim=1)    # [B, T, C] -> [2B, T, C]

    # Compute similarity matrix
    sim = torch.matmul(z, z.transpose(1, 2)) / supcon_temp  # B x T x T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    # Memory optimization for long sequences
    if seq_len > 1500:
        z, sim = z.cpu(), sim.cpu()
        torch.cuda.empty_cache()

    # Negative log-softmax loss
    logits = -F.log_softmax(logits, dim=-1)

    # Extract positive pair losses and average
    t = torch.arange(seq_len, device=embed_ori.device)
    loss = (logits[:, t, seq_len + t - 1].mean() + logits[:, seq_len + t, t].mean()) / 2
    return loss


def instance_supcon_loss(embed_ori, embed_aug, labels, supcon_temp=0.1):
    """
    Computes a sample-level contrastive loss with mixup augmentation.
    Encourages consistency between views along the batch dimension.
    
    Args:
        embed_ori (torch.Tensor): 视图 a 的嵌入, [B, T, C]
        embed_aug (torch.Tensor): 视图 b 的嵌入, [B, T, C]
        labels (torch.Tensor): 样本标签, [B]
        supcon_temp (float): 温度参数 (有监督模式通常建议设小一点，如 0.07)

    Returns:
        torch.Tensor: The computed sample contrastive loss.
    """
    B, T, C = embed_ori.size(0), embed_ori.size(1), embed_ori.size(2)
    device = embed_ori.device
    dtype = embed_ori.dtype
    labels = labels.to(device=device, dtype=torch.long).view(B)

    if B <= 1:
        return embed_ori.new_tensor(0.)

    z_ori = F.normalize(embed_ori, p=2, dim=-1)
    z_aug = F.normalize(embed_aug, p=2, dim=-1)

    # [2B, T, C] -> [T, 2B, C]，每个 t 上计算 (2B × 2B) 样本间相似度
    features = torch.cat([z_ori, z_aug], dim=0)
    z = features.transpose(0, 1).contiguous()
    sim_matrix = torch.bmm(z, z.transpose(1, 2)) / supcon_temp  # [T, 2B, 2B]

    labels_full = torch.cat([labels, labels], dim=0)
    mask = (labels_full.unsqueeze(0) == labels_full.unsqueeze(1)).to(dtype=dtype)
    logits_mask = torch.ones(2 * B, 2 * B, device=device, dtype=dtype)
    logits_mask.fill_diagonal_(0.0)
    mask = mask * logits_mask

    pos_count = mask.sum(dim=1)
    anchor_valid = pos_count > 0
    if not anchor_valid.any():
        return embed_ori.new_tensor(0.)

    logits_max, _ = torch.max(sim_matrix, dim=2, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    exp_logits = torch.exp(logits)
    logits_mask_t = logits_mask.unsqueeze(0).expand(T, -1, -1)
    log_prob = logits - torch.log((exp_logits * logits_mask_t).sum(dim=2, keepdim=True) + 1e-6)

    mean_log_prob_pos = (mask.unsqueeze(0) * log_prob).sum(dim=2) / (pos_count.unsqueeze(0) + 1e-6)
    loss = -mean_log_prob_pos[:, anchor_valid].mean(dim=1).mean()
    return loss


def subject_supcon_loss(embed_ori, embed_aug, subject_id, labels, supcon_temp=0.1):
    """
    Computes a subject-level contrastive loss with mixup augmentation.
    Mixup partners are drawn from **different** subjects when possible.
    Positive pairs are only among the first two blocks (embed_ori / embed_aug): same subject, not self.

    Args:
        embed_ori (torch.Tensor): Embeddings from view a, shape [B, T, C].
        embed_aug (torch.Tensor): Embeddings from view b, shape [B, T, C].
        subject_id (torch.Tensor): Subject IDs for each sample, shape [B].
        labels (torch.Tensor): Labels for each sample, shape [B].
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.

    Returns:
        torch.Tensor: The computed subject contrastive loss.
    """
    B, T, C = embed_ori.size(0), embed_ori.size(1), embed_ori.size(2)
    device = embed_ori.device
    dtype = embed_ori.dtype
    sub = subject_id.to(device=device, dtype=torch.long).view(B)

    if B <= 1:
        return embed_ori.new_tensor(0.)

    z_ori = F.normalize(embed_ori, p=2, dim=-1)
    z_aug = F.normalize(embed_aug, p=2, dim=-1)

    z = torch.cat([z_ori, z_aug], dim=0)
    z = z.transpose(0, 1).contiguous()
    sim = torch.bmm(z, z.transpose(1, 2)) / supcon_temp  # [T, 2B, 2B]

    sub2 = torch.cat([sub, sub], dim=0)
    mask = (sub2.unsqueeze(0) == sub2.unsqueeze(1)).to(dtype=dtype)
    mask = mask * (1.0 - torch.eye(2 * B, device=device, dtype=dtype))

    pos_count = mask.sum(dim=1)
    anchor_valid = pos_count > 0
    if not anchor_valid.any():
        return embed_ori.new_tensor(0.)

    logits_mask = torch.ones(2 * B, 2 * B, device=device, dtype=dtype)
    logits_mask.fill_diagonal_(0.0)
    logits_mask_t = logits_mask.unsqueeze(0).expand(T, -1, -1)

    sim_max, _ = torch.max(sim, dim=2, keepdim=True)
    logits = sim - sim_max.detach()
    exp_logits = torch.exp(logits) * logits_mask_t
    log_prob = logits - torch.log((exp_logits * logits_mask_t).sum(dim=2, keepdim=True) + 1e-6)

    mean_log_prob_pos = (mask.unsqueeze(0) * log_prob).sum(dim=2) / (pos_count.unsqueeze(0) + 1e-6)
    loss = -mean_log_prob_pos[:, anchor_valid].mean(dim=1).mean()
    return loss


def hierarchical_supcon_loss(
    embed_ori, embed_aug, labels, subject_id,
    lambda_temporal=0.0, lambda_instance=0.0, lambda_subject=0.0,
    temporal_unit=0, supcon_temp=1.0):
    """
    Computes a hierarchical contrastive loss across multiple temporal scales.
    Combines instance-level, temporal-level and optionally subject-level contrastive losses at each scale.

    Args:
        embed_ori (torch.Tensor): Embeddings from view a, shape [B, T, C].
        embed_aug (torch.Tensor): Embeddings from view b, shape [B, T, C].
        labels (torch.Tensor): Labels for each sample, shape [B].
        subject_id (torch.Tensor, optional): Subject IDs for each sample, shape [B]. If None, subject loss is skipped.
        lambda_temporal (float, optional): Weight for temporal contrastive loss. Defaults to 0.0.
        lambda_instance (float, optional): Weight for instance contrastive loss. Defaults to 0.0.
        lambda_subject (float, optional): Weight for subject contrastive loss. Defaults to 0.0.
        temporal_unit (int, optional): Minimum depth to include temporal loss. Defaults to 0.
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.

    Returns:
        tuple: (instance_loss, temporal_loss, subject_loss, total_loss), each is a scalar torch.Tensor.
    """
    loss_instance = torch.tensor(0., device=embed_ori.device)
    loss_temporal = torch.tensor(0., device=embed_ori.device)
    loss_subject = torch.tensor(0., device=embed_ori.device)
    d_temp, d_inst = 0, 0
    downsamplRate = 4

    while embed_ori.size(1) > downsamplRate:
        # Temporal-level contrastive loss
        if lambda_temporal != 0 and d_temp >= temporal_unit:
            loss_temporal += temporal_supcon_loss(embed_ori, embed_aug, supcon_temp=supcon_temp)
        # Instance-level contrastive loss
        if lambda_instance != 0:
            loss_instance += instance_supcon_loss(embed_ori, embed_aug, labels, supcon_temp=supcon_temp)
        # Subject-level contrastive loss
        if lambda_subject != 0:
            loss_subject += subject_supcon_loss(embed_ori, embed_aug, subject_id, labels, supcon_temp=supcon_temp)
        d_temp += 1
        d_inst += 1

        # embed_ori = F.max_pool1d(embed_ori.transpose(1, 2), kernel_size=downsamplRate).transpose(1, 2)
        # embed_aug = F.max_pool1d(embed_aug.transpose(1, 2), kernel_size=downsamplRate).transpose(1, 2)
        embed_ori = F.avg_pool1d(embed_ori.transpose(1, 2), kernel_size=downsamplRate).transpose(1, 2)
        embed_aug = F.avg_pool1d(embed_aug.transpose(1, 2), kernel_size=downsamplRate).transpose(1, 2)

    # Handle final single-frame case
    if embed_ori.size(1) <= downsamplRate:
        if lambda_instance != 0:
            loss_instance += instance_supcon_loss(embed_ori, embed_aug, labels, supcon_temp=supcon_temp)
        if lambda_subject != 0:
            loss_subject += subject_supcon_loss(embed_ori, embed_aug, subject_id, labels, supcon_temp=supcon_temp)
        d_inst += 1

    # Normalize by depth and compute total
    loss_temporal = loss_temporal / d_temp
    loss_instance = loss_instance / d_inst
    loss_subject = loss_subject / d_inst
    loss_total = lambda_temporal * loss_temporal + lambda_instance * loss_instance + lambda_subject * loss_subject
    return loss_temporal, loss_instance, loss_subject, loss_total


def hierarchical_supcon_loss_multiscale(
    embed_ori, embed_aug, labels, subject_id, 
    lambda_temporal=0.0, lambda_instance=0.0, lambda_subject=0.0, 
    temporal_unit=0, supcon_temp=1.0, layer_exp_base=0.5):
    """
    Computes a hierarchical contrastive loss across multiple temporal scales.
    Combines instance-level, temporal-level and optionally subject-level contrastive losses at each scale.

    Args:
        embed_ori (torch.Tensor or tuple/list): View a: single tensor [B, T, C], or per-layer tensors from encoder.
        embed_aug (torch.Tensor or tuple/list): View b, same structure as embed_ori.
        labels (torch.Tensor): Labels for each sample, shape [B].
        subject_id (torch.Tensor, optional): Subject IDs for each sample, shape [B]. If None, subject loss is skipped.
        lambda_temporal (float, optional): Weight for temporal contrastive loss. Defaults to 0.0.
        lambda_instance (float, optional): Weight for instance contrastive loss. Defaults to 0.0.
        lambda_subject (float, optional): Weight for subject contrastive loss. Defaults to 0.0.
        temporal_unit (int, optional): Minimum layer index to include temporal loss. Defaults to 0.
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
        layer_exp_base (float, optional): Decay factor in ``(0, 1]`` (from shallow toward deep).
            Layer weights are ``w_l = base ** (L - 1 - l)`` with shallow ``l=0`` and deep ``l=L-1``,
            so shallower layers get smaller weights and deeper layers larger.
            ``1.0`` means uniform weights (no skew).

    Returns:
        tuple: (temporal_loss, instance_loss, subject_loss, total_loss), each is a scalar torch.Tensor.
    """
    if not (0.0 < layer_exp_base <= 1.0):
        raise ValueError("layer_exp_base must be in (0, 1], e.g. 0.9 for gentle skew toward deep layers.")

    if not isinstance(embed_ori, (tuple, list)):
        embed_ori = (embed_ori,)
        embed_aug = (embed_aug,)
    ref = embed_ori[0]
    loss_temporal = ref.new_tensor(0.)
    loss_instance = ref.new_tensor(0.)
    loss_subject = ref.new_tensor(0.)
    n_layers = len(embed_ori)
    if n_layers == 0:
        loss_total = loss_temporal + loss_instance + loss_subject
        return loss_temporal, loss_instance, loss_subject, loss_total

    # Shallow l=0 -> base^(L-1) (smallest when base<1); deep l=L-1 -> 1 (largest).
    layer_weights = [
        layer_exp_base ** (n_layers - 1 - l) for l in range(n_layers)
    ]
    weight_sum = float(sum(layer_weights))

    for layer_idx, (z_ori, z_aug) in enumerate(zip(embed_ori, embed_aug)):
        w = layer_weights[layer_idx]
        if lambda_temporal != 0 and layer_idx >= temporal_unit:
            loss_temporal = loss_temporal + w * temporal_supcon_loss(z_ori, z_aug, supcon_temp=supcon_temp)
        if lambda_instance != 0:
            loss_instance = loss_instance + w * instance_supcon_loss(z_ori, z_aug, labels, supcon_temp=supcon_temp)
        if lambda_subject != 0:
            loss_subject = loss_subject + w * subject_supcon_loss(z_ori, z_aug, subject_id, labels, supcon_temp=supcon_temp)

    if lambda_temporal != 0:
        loss_temporal = loss_temporal / weight_sum
    if lambda_instance != 0:
        loss_instance = loss_instance / weight_sum
    if lambda_subject != 0:
        loss_subject = loss_subject / weight_sum

    loss_total = (
        lambda_temporal * loss_temporal
        + lambda_instance * loss_instance
        + lambda_subject * loss_subject
    )
    return loss_temporal, loss_instance, loss_subject, loss_total


#%% 受试者解码结果 KL散度 wasserstein 正则化
def kl_to_subject_mean_(pred_logits, true_labels, temperature=1.0, detach_mean=True, eps=1e-6):
    """
    按受试者分组，计算组内预测分布到组内均值的 KL 散度。

    Args:
        pred_logits: 预测 logits，形状 (sample, n_classes)
        true_labels: 形状 (sample, 2)，第一列为真实类别标签，第二列为受试者 id；用于得到 B、K 分组
        temperature: softmax 温度
        detach_mean: 是否对组内均值 detach
        eps: 数值稳定常数

    Returns:
        组内 KL 散度的均值（标量）
    """
    device = pred_logits.device
    subject_ids = true_labels[:, 1]  # (sample,)
    unique_subjects = torch.unique(subject_ids)

    total_kl = 0.0
    total_count = 0

    for i in range(len(unique_subjects)):
        s = unique_subjects[i]
        mask = (subject_ids == s)
        indices = mask.nonzero(as_tuple=True)[0]
        z = pred_logits[indices]  # (K_i, C)
        K_i = z.size(0)
        if K_i < 2:
            continue
        C = z.size(-1)
        p = F.softmax(z / temperature, dim=-1).clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True)
        logp = torch.log(p)
        p_bar = p.mean(dim=0, keepdim=True).clamp_min(eps)  # (1, C)
        p_bar = p_bar / p_bar.sum(dim=-1, keepdim=True)
        if detach_mean:
            p_bar = p_bar.detach()
        logp_bar = torch.log(p_bar)
        kl_i = (p * (logp - logp_bar)).sum(dim=-1)  # (K_i,)
        total_kl = total_kl + kl_i.sum()
        total_count = total_count + K_i

    if total_count == 0:
        return torch.tensor(0.0, device=device)
    return total_kl / total_count


def kl_reg_grouo_mean(
    pred_logits,
    group_ids,
    temperature=0.1,
    detach_mean=True,
    eps=1e-6,
    reduction="subject",       # "sample" or "subject"
):
    """
    KL(p_i || p_bar_subject(i)) consistency regularization.
    Returns a scalar.
    """
    device = pred_logits.device
    N, C = pred_logits.shape

    uniq, inv = torch.unique(group_ids, return_inverse=True)  # inv: (N,)
    B = uniq.numel()

    # p: (N,C)
    p = F.softmax(pred_logits / temperature, dim=-1).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)

    # sum_p per subject: (B,C)
    sum_p = torch.zeros(B, C, device=device, dtype=p.dtype)
    sum_p.index_add_(0, inv, p)

    # counts per subject: (B,1)
    counts = torch.bincount(inv, minlength=B).to(device=device).unsqueeze(1)
    counts_safe = counts.clamp_min(1)

    # mean distribution per subject: (B,C)
    p_bar = (sum_p / counts_safe).clamp_min(eps)
    p_bar = p_bar / p_bar.sum(dim=-1, keepdim=True)

    if detach_mean:
        p_bar = p_bar.detach()

    # map subject mean back to each sample: (N,C)
    p_bar_per_sample = p_bar[inv]

    # KL per sample: (N,)
    kl = (p * (torch.log(p) - torch.log(p_bar_per_sample))).sum(dim=-1)

    # optionally drop subjects with <2 samples
    valid_subject = (counts.squeeze(1) >= 2)            # (B,)
    valid_mask = valid_subject[inv]                     # (N,)
    if valid_mask.any():
        kl = kl[valid_mask]
        inv_valid = inv[valid_mask]
    else:
        return torch.tensor(0.0, device=device)

    if reduction == "sample":
        # Each sample equally weighted (like CE default mean over batch)
        return kl.mean()

    elif reduction == "subject":
        # Each subject equally weighted:
        # mean KL per subject, then average subjects
        sum_kl = torch.zeros(B, device=device, dtype=kl.dtype)
        cnt_kl = torch.zeros(B, device=device, dtype=kl.dtype)
        sum_kl.index_add_(0, inv_valid, kl)
        cnt_kl.index_add_(0, inv_valid, torch.ones_like(kl))
        subj_mean_kl = sum_kl / cnt_kl.clamp_min(1)

        return subj_mean_kl[valid_subject].mean()

    else:
        raise ValueError("reduction must be 'sample' or 'subject'")


def wasserstein_reg_group_mean(
        pred_logits,
        group_ids,
        temperature=1.0,
        detach_mean=True,
        reduction="subject",  # "sample" or "subject"
        p_norm=2,  # Wasserstein 阶数 (p=2 为常见的 W2 距离)
):
    """
    Wasserstein 一致性正则化（按受试者均值分布）。

    Note:
        采用类别轴(0..C-1)上一维离散分布的闭式计算：
            W_p(p, q) = (sum_k |CDF_p(k) - CDF_q(k)|^p)^(1/p)
        返回每个样本对应受试者均值分布的距离，形状为 (N,)。
        参数 ``blur`` 保留用于接口兼容（该实现未使用）。
    """
    device = pred_logits.device
    N, C = pred_logits.shape

    # 1. 提取受试者 ID 和映射关系
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    B = uniq.numel()

    # 2. 转化为概率分布 (p: N, C)
    p = F.softmax(pred_logits / temperature, dim=-1)

    # 3. 计算受试者平均分布 (p_bar: B, C)
    sum_p = torch.zeros(B, C, device=device, dtype=p.dtype)
    sum_p.index_add_(0, inv, p)
    counts = torch.bincount(inv, minlength=B).to(device=device).float().unsqueeze(1)
    counts_safe = counts.clamp_min(1)
    p_bar = sum_p / counts_safe

    if detach_mean:
        p_bar = p_bar.detach()

    # 4. 映射回每个样本 (p_bar_per_sample: N, C)
    p_bar_per_sample = p_bar[inv]

    # 5. 在类别轴上计算逐样本 Wasserstein 距离（N,）
    p_norm = float(p_norm)
    if p_norm <= 0:
        raise ValueError("p_norm must be > 0")
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(p_bar_per_sample, dim=-1)
    w_dist = torch.sum(torch.abs(cdf_p - cdf_q).pow(p_norm), dim=-1).pow(1.0 / p_norm)

    # 7. 过滤样本数不足 2 的受试者
    valid_subject = (counts.squeeze(1) >= 2)
    valid_mask = valid_subject[inv]
    if not valid_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    w_dist = w_dist[valid_mask]
    inv_valid = inv[valid_mask]

    # 8. Reduction 逻辑
    if reduction == "sample":
        return w_dist.mean()
    elif reduction == "subject":
        sum_w = torch.zeros(B, device=device, dtype=w_dist.dtype)
        cnt_w = torch.zeros(B, device=device, dtype=w_dist.dtype)
        sum_w.index_add_(0, inv_valid, w_dist)
        cnt_w.index_add_(0, inv_valid, torch.ones_like(w_dist))
        subj_mean_w = sum_w / cnt_w.clamp_min(1)
        return subj_mean_w[valid_subject].mean()
    else:
        raise ValueError("reduction must be 'sample' or 'subject'")


def kl_reg_group_mean_multiscale(
    layer_out,
    logits=None,
    subject_ids=None,
    sample_ids=None,
    lambda_ss=0.5,
    temperature=1.0,
    detach_mean=True,
    eps=1e-6,
    reduction="subject",
    layer_exp_base=0.5,
):
    """
    Computes a hierarchical multi-layer KL regularization to group means.

    For each layer logits ``z_l``, computes:
        ``KL(p_l || pbar_group(l))``
    via ``kl_reg_grouo_mean``, then applies the same layer weighting strategy as
    ``crossentropy_loss_multiscale`` / ``hierarchical_supcon_loss_multiscale``:
        shallow l=0 -> base^(L-1) (smallest when base<1)
        deep    l=L-1 -> 1        (largest)

    Args:
        layer_out: tuple/list of intermediate-layer logits, or None.
        logits: final-layer logits tensor, can be None.
        subject_ids: Subject-level group ids, shape [N] or [N, 1].
        sample_ids: Sample/session-level group ids, shape [N] or [N, 1].
        lambda_ss: Weight for subject term when both ids are provided.
            Combined loss = lambda_ss * loss_subject + (1 - lambda_ss) * loss_sample.
        temperature: Softmax temperature for KL regularization.
        detach_mean: Whether to detach per-subject mean distribution.
        eps: Numerical stability constant passed to ``kl_reg_subject_mean``.
        reduction: "sample" or "subject", passed to ``kl_reg_subject_mean``.
        layer_exp_base: Layer weighting base in (0, 1].

    Returns:
        torch.Tensor: scalar KL regularization term.
    """
    if not (0.0 < layer_exp_base <= 1.0):
        raise ValueError(
            "layer_exp_base must be in (0, 1], e.g. 0.9 for gentle skew toward deep layers."
        )
    if subject_ids is None and sample_ids is None:
        raise ValueError("At least one of subject_ids or sample_ids must be provided.")
    if subject_ids is not None and sample_ids is not None and not (0.0 <= lambda_ss <= 1.0):
        raise ValueError("lambda_ss must be in [0, 1] when both subject_ids and sample_ids are provided.")

    if layer_out is None:
        layer_out = ()
    elif not isinstance(layer_out, (tuple, list)):
        raise TypeError("layer_out must be tuple/list (or None).")

    all_logits = tuple(layer_out)
    if logits is not None:
        all_logits = all_logits + (logits,)

    n_layers = len(all_logits)
    if n_layers == 0:
        return torch.tensor(0.0)

    ref = all_logits[0]
    loss_kl = ref.new_tensor(0.0)

    layer_weights = [
        layer_exp_base ** (n_layers - 1 - l) for l in range(n_layers)
    ]
    weight_sum = float(sum(layer_weights))

    for layer_idx, z in enumerate(all_logits):
        w = layer_weights[layer_idx]
        if subject_ids is not None and sample_ids is not None:
            loss_subject = kl_reg_grouo_mean(
                pred_logits=z,
                group_ids=subject_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                eps=eps,
                reduction=reduction,
            )
            loss_sample = kl_reg_grouo_mean(
                pred_logits=z,
                group_ids=sample_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                eps=eps,
                reduction=reduction,
            )
            loss_layer = (1-lambda_ss) * loss_subject + lambda_ss * loss_sample
        elif subject_ids is not None:
            loss_layer = kl_reg_grouo_mean(
                pred_logits=z,
                group_ids=subject_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                eps=eps,
                reduction=reduction,
            )
        else:
            loss_layer = kl_reg_grouo_mean(
                pred_logits=z,
                group_ids=sample_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                eps=eps,
                reduction=reduction,
            )
        loss_kl = loss_kl + w * loss_layer

    loss_kl = loss_kl / weight_sum
    return loss_kl


def wasserstein_reg_group_mean_multiscale(
    layer_out,
    logits=None,
    subject_ids=None,
    sample_ids=None,
    lambda_ss=0.5,
    temperature=1.0,
    detach_mean=True,
    reduction="subject",
    p_norm=2,
    layer_exp_base=0.5,
):
    """
    Computes a hierarchical multi-layer Wasserstein regularization to group means.

    For each layer logits ``z_l``, computes:
        ``W(z_l, zbar_group(l))``
    via ``wasserstein_reg_group_mean``, then applies the same layer weighting
    strategy as other multiscale losses:
        shallow l=0 -> base^(L-1) (smallest when base<1)
        deep    l=L-1 -> 1        (largest)

    Args:
        layer_out: tuple/list of intermediate-layer logits, or None.
        logits: final-layer logits tensor, can be None.
        subject_ids: Subject-level group ids, shape [N] or [N, 1].
        sample_ids: Sample/session-level group ids, shape [N] or [N, 1].
        lambda_ss: Weight for subject term when both ids are provided.
            Combined loss = lambda_ss * loss_subject + (1 - lambda_ss) * loss_sample.
        temperature: Softmax temperature.
        detach_mean: Whether to detach per-subject mean distribution.
        reduction: "sample" or "subject", passed to single-layer function.
        p_norm: Wasserstein order for Sinkhorn in geomloss.
        layer_exp_base: Layer weighting base in (0, 1].

    Returns:
        torch.Tensor: scalar Wasserstein regularization term.
    """
    if not (0.0 < layer_exp_base <= 1.0):
        raise ValueError(
            "layer_exp_base must be in (0, 1], e.g. 0.9 for gentle skew toward deep layers."
        )
    if subject_ids is None and sample_ids is None:
        raise ValueError("At least one of subject_ids or sample_ids must be provided.")
    if subject_ids is not None and sample_ids is not None and not (0.0 <= lambda_ss <= 1.0):
        raise ValueError("lambda_ss must be in [0, 1] when both subject_ids and sample_ids are provided.")

    if layer_out is None:
        layer_out = ()
    elif not isinstance(layer_out, (tuple, list)):
        raise TypeError("layer_out must be tuple/list (or None).")

    all_logits = tuple(layer_out)
    if logits is not None:
        all_logits = all_logits + (logits,)

    n_layers = len(all_logits)
    if n_layers == 0:
        return torch.tensor(0.0)

    ref = all_logits[0]
    loss_w = ref.new_tensor(0.0)

    layer_weights = [
        layer_exp_base ** (n_layers - 1 - l) for l in range(n_layers)
    ]
    weight_sum = float(sum(layer_weights))

    for layer_idx, z in enumerate(all_logits):
        w = layer_weights[layer_idx]
        if subject_ids is not None and sample_ids is not None:
            loss_subject = wasserstein_reg_group_mean(
                pred_logits=z,
                group_ids=subject_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                reduction=reduction,
                p_norm=p_norm,
            )
            loss_sample = wasserstein_reg_group_mean(
                pred_logits=z,
                group_ids=sample_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                reduction=reduction,
                p_norm=p_norm,
            )
            loss_layer = lambda_ss * loss_subject + (1.0 - lambda_ss) * loss_sample
        elif subject_ids is not None:
            loss_layer = wasserstein_reg_group_mean(
                pred_logits=z,
                group_ids=subject_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                reduction=reduction,
                p_norm=p_norm,
            )
        else:
            loss_layer = wasserstein_reg_group_mean(
                pred_logits=z,
                group_ids=sample_ids.view(-1),
                temperature=temperature,
                detach_mean=detach_mean,
                reduction=reduction,
                p_norm=p_norm,
            )
        loss_w = loss_w + w * loss_layer

    loss_w = loss_w / weight_sum
    return loss_w


#%% 多层加权损失
def crossentropy_loss_multiscale(
    layer_out, logits,
    targets,
    criterion,
    layer_exp_base=0.5,
):
    """
    Computes a hierarchical multi-layer Cross-Entropy loss.

    This function only computes CE losses (no contrastive/temporal/subject components).
    Inputs are:
      - layer_out: tuple/list of intermediate-layer logits
      - logits: final-layer logits tensor

    The per-layer weight follows the same logic as in
    ``hierarchical_supcon_loss_multiscale``:
        shallow l=0 -> base^(L-1)  (smallest when base<1)
        deep    l=L-1 -> 1        (largest)

    Returns:
        torch.Tensor: scalar CE loss (unless ``reduction="none"``, then returns a tensor).
    """
    if not (0.0 < layer_exp_base <= 1.0):
        raise ValueError(
            "layer_exp_base must be in (0, 1], e.g. 0.9 for gentle skew toward deep layers."
        )

    if layer_out is None:
        layer_out = ()
    elif not isinstance(layer_out, (tuple, list)):
        raise TypeError("layer_out must be tuple/list (or None).")

    all_logits = tuple(layer_out) + (logits,)
    ref = all_logits[0]
    loss_ce = ref.new_tensor(0.0)
    n_layers = len(all_logits)
    if n_layers == 0:
        return loss_ce

    # Shallow l=0 -> base^(L-1) (smallest when base<1); deep l=L-1 -> 1 (largest).
    layer_weights = [
        layer_exp_base ** (n_layers - 1 - l) for l in range(n_layers)
    ]
    weight_sum = float(sum(layer_weights))

    # cross_entropy expects targets as int64 class indices
    targets = targets.to(device=ref.device, dtype=torch.long)

    for layer_idx, z in enumerate(all_logits):
        w = layer_weights[layer_idx]
        loss_layer = criterion(z, targets)
        loss_ce = loss_ce + w * loss_layer

    # Normalize by the sum of layer weights to keep scale stable with different L.
    loss_ce = loss_ce / weight_sum
    return loss_ce


#%% 有监督对比损失，数据增强版本，不是两个视角的版本
def instance_supcon_loss_universum_augmentation(
    embed, labels, supcon_temp=0.1, supcon_alpha=0.2, universum=False, perm_indices=None
):
    """
    Computes a sample-level contrastive loss with mixup augmentation.
    Encourages consistency between views along the batch dimension.

    Args:
        embed (torch.Tensor): 视图 a 的嵌入, [B, C]
        labels (torch.Tensor): 样本标签, [B]
        supcon_temp (float): 温度参数 (有监督模式通常建议设小一点，如 0.07)
        supcon_alpha (float): Mixup 系数
        universum (bool): 是否使用 universum augmentation(默认 False)
        perm_indices (torch.Tensor, optional): Mixup indices, shape [B]. If None, perm_indices will be computed.
    """
    B, C = embed.size(0), embed.size(1)
    device, dtype = embed.device, embed.dtype
    labels = labels.to(device=device, dtype=torch.long).view(B)

    if B <= 1:
        return embed.new_tensor(0.)

    if universum:
        if perm_indices is None:
            with torch.no_grad():
                perm_indices = _mixup_partner_diff_label(labels, B, device)
        uni = supcon_alpha * embed + (1 - supcon_alpha) * embed[perm_indices, :]
        uni_norm = F.normalize(uni, p=2, dim=-1)

    embed_norm = F.normalize(embed, p=2, dim=-1)

    if universum:
        features = torch.cat([embed_norm, uni_norm], dim=0).contiguous()
    else:
        features = embed_norm.contiguous()
    sim_matrix = torch.matmul(features, features.T) / supcon_temp  # [B, B]

    labels_view = labels.contiguous().view(-1, 1) # [1, B]
    mask = (labels_view == labels_view.T).to(dtype=dtype) # [B, B]，相同标签的位置为1
    logits_mask = torch.ones_like(mask) # [B, B]，全1的矩阵
    logits_mask.fill_diagonal_(0) # [B, B]，全1矩阵对角线设置为0
    mask = mask * logits_mask # [B, B]，将对角线位置设置为0

    logits_max, _ = torch.max(sim_matrix[:B, :], dim=1, keepdim=True) # [1, B]或者[1, 2B]，[B, B]原始数据的计算结果，[B, B:2B]uni与原始数据的计算结果
    logits = sim_matrix[:B, :] - logits_max.detach() # [B, B]或者[B, 2B]
    exp_logits = torch.exp(logits) # [B, B]或者[B, 2B]

    if universum:
        full_logits_mask = torch.ones((B, 2 * B), device=device, dtype=dtype) # [B, 2B]
        full_logits_mask[:, :B] = logits_mask # [B, 2B]
    else:
        full_logits_mask = logits_mask # [B, B]

    log_prob = logits - torch.log((exp_logits * full_logits_mask).sum(1, keepdim=True) + 1e-6) # [B, B]或者[B, 2B] - [B, 1]

    mean_log_prob_pos = (mask * log_prob[:, :B]).sum(1) / (mask.sum(1) + 1e-6) # [B, 1]
    loss = -mean_log_prob_pos.mean()  # [1]
    return loss


def subject_supcon_loss_universum_augmentation(
    embed, subject_id, labels, supcon_temp=0.1, supcon_alpha=0.2, universum=False, partner=None
):
    """
    Computes a subject-level contrastive loss with mixup augmentation.
    Mixup partners are drawn from **different** subjects when possible.
    Positive pairs are only among the first two blocks (embed_ori / embed_aug): same subject, not self.

    Args:
        embed (torch.Tensor): Embeddings from view a, shape [B, C].
        subject_id (torch.Tensor): Subject IDs for each sample, shape [B].
        labels (torch.Tensor): Labels for each sample, shape [B].
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
        supcon_alpha (float, optional): Mixup coefficient for Universum. Defaults to 0.2.
        universum (bool, optional): Whether to use Universum augmentation. Defaults to False.
        partner (torch.Tensor, optional): Mixup indices, shape [B]. If None, partner will be computed.

    Returns:
        torch.Tensor: The computed subject contrastive loss.
    """
    B, C = embed.size(0), embed.size(1)
    device, dtype = embed.device, embed.dtype
    sub = subject_id.to(device=device, dtype=torch.long).view(B)
    labels = labels.to(device=device, dtype=torch.long).view(B)

    if B <= 1:
        return embed.new_tensor(0.)

    if universum:
        if partner is None:
            with torch.no_grad():
                partner = _mixup_partners_diff_subject_label(sub, labels, B, device)
        uni = supcon_alpha * embed + (1 - supcon_alpha) * embed[partner, :]
        uni_norm = F.normalize(uni, p=2, dim=-1)

    embed_norm = F.normalize(embed, p=2, dim=-1)

    if universum:
        features = torch.cat([embed_norm, uni_norm], dim=0).contiguous()
    else:
        features = embed_norm.contiguous()
    sim_matrix = torch.matmul(features, features.T) / supcon_temp  # [B, B]

    sub_view = sub.contiguous().view(-1, 1) # [1, B]
    labels_view = labels.contiguous().view(-1, 1) # [1, B]

    mask_label = (labels_view == labels_view.T).to(dtype=dtype) # [B, B]，相同标签的位置为1
    mask_diff_sub = (sub_view != sub_view.T).to(dtype=dtype)  # [B, B]，不同受试者标签的位置为1
    mask_same_sub = (sub_view == sub_view.T).to(dtype=dtype) # [B, B]，相同受试者标签的位置为1
    mask = mask_label * mask_diff_sub # [B, B]，相同标签且不同受试者标签的位置为1
    ignore_mask = mask_label * mask_same_sub # [B, B]，相同标签且相同受试者标签的位置为1
    # mask = (sub_view == sub_view.T).to(dtype=dtype) # [B, B]，相同受试者标签的位置为1

    logits_mask = torch.ones_like(mask) # [B, B]，全1的矩阵
    logits_mask.fill_diagonal_(0) # [B, B]，全1矩阵对角线设置为0
    mask = mask * logits_mask # [B, B]，将对角线位置设置为0

    logits_max, _ = torch.max(sim_matrix[:B, :], dim=1, keepdim=True) # [B, 1, 1]或者[2B, 1, 1]，[B, B]原始数据的计算结果，[2B, B]uni与原始数据的计算结果
    logits = sim_matrix[:B, :] - logits_max.detach() # [B, B]或者[2B, B]
    exp_logits = torch.exp(logits) # [B, B]或者[2B, B]

    original_logits_mask = logits_mask * (1.0 - ignore_mask)
    if universum:
        full_logits_mask = torch.ones((B, 2 * B), device=device, dtype=dtype) # [B, 2B]
        full_logits_mask[:, :B] = original_logits_mask # [B, 2B]
    else:
        full_logits_mask = original_logits_mask # [B, B]

    log_prob = logits - torch.log((exp_logits * full_logits_mask).sum(1, keepdim=True) + 1e-6) # [B, B]或者[B, 2B] - [B, 1]

    mean_log_prob_pos = (mask * log_prob[:, :B]).sum(1) / (mask.sum(1) + 1e-6) # [B, 1]
    loss = -mean_log_prob_pos.mean()  # [1]
    return loss


def hierarchical_supcon_loss_multiscale_augmentation(
    embed, labels, subject_id, lambda_instance=0.0, lambda_subject=0.0,
    supcon_temp=1.0, supcon_alpha=0.2, layer_exp_base=0.5, universum=False
):
    """
    Computes a hierarchical contrastive loss across multiple temporal scales.
    Combines instance-level, temporal-level and optionally subject-level contrastive losses at each scale.

    Args:
        embed (torch.Tensor or tuple/list): View a: single tensor [B, C], or per-layer tensors from encoder.
        labels (torch.Tensor): Labels for each sample, shape [B].
        subject_id (torch.Tensor, optional): Subject IDs for each sample, shape [B]. If None, subject loss is skipped.
        lambda_temporal (float, optional): Weight for temporal contrastive loss. Defaults to 0.0.
        lambda_instance (float, optional): Weight for instance contrastive loss. Defaults to 0.0.
        lambda_subject (float, optional): Weight for subject contrastive loss. Defaults to 0.0.
        temporal_unit (int, optional): Minimum layer index to include temporal loss. Defaults to 0.
        supcon_temp (float, optional): Temperature parameter for contrastive loss. Defaults to 1.0.
        layer_exp_base (float, optional): Decay factor in ``(0, 1]`` (from shallow toward deep).
            Layer weights are ``w_l = base ** (L - 1 - l)`` with shallow ``l=0`` and deep ``l=L-1``,
            so shallower layers get smaller weights and deeper layers larger.
            ``1.0`` means uniform weights (no skew).
        universum (bool, optional): Whether to use Universum augmentation. Defaults to False.

    Returns:
        tuple: (temporal_loss, instance_loss, subject_loss, total_loss), each is a scalar torch.Tensor.
    """
    if not (0.0 < layer_exp_base <= 1.0):
        raise ValueError("layer_exp_base must be in (0, 1], e.g. 0.9 for gentle skew toward deep layers.")

    if not isinstance(embed, (tuple, list)):
        embed = (embed,)
    ref = embed[0]
    loss_instance = ref.new_tensor(0.)
    loss_subject = ref.new_tensor(0.)
    n_layers = len(embed)
    if n_layers == 0:
        loss_total = loss_instance + loss_subject
        return loss_instance, loss_subject, loss_total

    # Shallow l=0 -> base^(L-1) (smallest when base<1); deep l=L-1 -> 1 (largest).
    layer_weights = [
        layer_exp_base ** (n_layers - 1 - l) for l in range(n_layers)
    ]
    weight_sum = float(sum(layer_weights))

    if universum:
        with torch.no_grad():
            perm_indices = _mixup_partner_diff_label(labels, embed.size(0), embed.device)
            partner = _mixup_partners_diff_subject_label(subject_id, labels, embed.size(0), embed.device)
    else:
        perm_indices, partner = None, None

    for layer_idx, z in enumerate(embed):
        w = layer_weights[layer_idx]
        if lambda_instance != 0:
            loss_instance = loss_instance + w * instance_supcon_loss_universum_augmentation(z, labels, supcon_temp, supcon_alpha, universum, perm_indices)
        if lambda_subject != 0:
            loss_subject = loss_subject + w * subject_supcon_loss_universum_augmentation(z, subject_id, labels, supcon_temp, supcon_alpha, universum, partner)

    if lambda_instance != 0:
        loss_instance = loss_instance / weight_sum
    if lambda_subject != 0:
        loss_subject = loss_subject / weight_sum

    loss_total = lambda_instance * loss_instance + lambda_subject * loss_subject

    return loss_instance, loss_subject, loss_total