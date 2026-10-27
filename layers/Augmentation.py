import torch
import torch.nn as nn
import numpy as np



class Jitter(nn.Module):
    # apply noise on each element
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x) * self.scale
        return x


class Scale(nn.Module):
    # scale each channel by a random scalar
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            x *= 1 + torch.randn(B, C, 1, device=x.device) * self.scale
        return x


class Flip(nn.Module):
    # left-right flip
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            return torch.flip(x, [-1])
        return x


class Shuffle(nn.Module):
    # shuffle channels order
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            B, C, T = x.shape
            perm = torch.randperm(C)
            return x[:, perm, :]
        return x


class TemporalMask(nn.Module):
    # Randomly mask a portion of timestamps across all channels
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(T * self.ratio)
            mask_indices = torch.randperm(T)[:num_mask]
            x[:, :, mask_indices] = 0
        return x


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            # Perform rfft
            x_fft = torch.fft.rfft(x, dim=-1)
            # Generate random indices for masking
            mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
            # Apply mask
            x_fft = x_fft * mask
            # Perform inverse rfft
            x = torch.fft.irfft(x_fft, n=T, dim=-1)
        return x


class ChannelMask(nn.Module):
    # Randomly mask a portion of channels across all timestamps
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(C * self.ratio)
            mask_indices = torch.randperm(C)[:num_mask]
            x[:, mask_indices, :] = 0
        return x


#%% 数据增强
def _partner_same_class_diff_subject(label, subject_id, device, generator=None):
    """For each index i, pick random j with same class and different subject; else j=i."""
    B = label.shape[0]
    lab = label.to(device).view(-1).long()
    sub = subject_id.to(device).view(-1).long()
    partner = torch.arange(B, device=device, dtype=torch.long)
    for i in range(B):
        mask = (lab == lab[i]) & (sub != sub[i])
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx.numel() > 0:
            r = torch.randint(
                0,
                idx.numel(),
                (1,),
                device=device,
                generator=generator,
            )
            partner[i] = idx[r]
    return partner


def _partner_diff_class(
    label,
    subject_id,
    device,
    generator=None,
    *,
    prefer_diff_subject=True,
):
    """
    For each index i, pick random j with different class; else j=i.

    If ``prefer_diff_subject`` is True, it first tries (different class AND different subject),
    and falls back to (different class) when no such sample exists.
    """
    B = label.shape[0]
    lab = label.to(device).view(-1).long()
    sub = subject_id.to(device).view(-1).long()
    partner = torch.arange(B, device=device, dtype=torch.long)
    for i in range(B):
        if prefer_diff_subject:
            mask = (lab != lab[i]) & (sub != sub[i])
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            if idx.numel() == 0:
                mask = (lab != lab[i])
                idx = torch.nonzero(mask, as_tuple=False).view(-1)
        else:
            mask = (lab != lab[i])
            idx = torch.nonzero(mask, as_tuple=False).view(-1)

        if idx.numel() > 0:
            r = torch.randint(
                0,
                idx.numel(),
                (1,),
                device=device,
                generator=generator,
            )
            partner[i] = idx[r]
    return partner


def frequency_mixing(
    batch_x,
    rate=0.5,
    dim=1,
    label=None,
    subject_id=None,
    seed=None,
    pairing_mode="train",
    train_pairing="same_class",
):
    """
    Frequency-domain mixup augmentation. Mixes amplitude spectra between samples
    while preserving dominant frequencies, then reconstructs the signal.

    When ``label`` and ``subject_id`` are both provided, partner indices depend on
    ``pairing_mode`` and ``train_pairing``:

    - ``"val"``: same subject, different row in the batch (other sample of that subject).
    - otherwise (e.g. ``"train"``): controlled by ``train_pairing``:
        - ``"same_class"`` (default): same class, different subject (``_partner_same_class_diff_subject``).
        - ``"diff_class"``: different class (优先不同被试；若不存在则退化为仅不同类别).
        - ``"both"``: 同时执行上述两种训练配对策略，并在 batch 维拼接输出（输出 batch 大小变为 2B）。

    If no valid partner exists for a row, it pairs with itself.

    Args:
        batch_x (torch.Tensor): Input batch, shape [B, ...].
        rate (float, optional): Mixup rate for mask. Defaults to 0.5.
        dim (int, optional): Dimension along which FFT is applied. Defaults to 1.
        label (torch.Tensor, optional): Class id per sample, shape [B].
        subject_id (torch.Tensor, optional): Subject id per sample, shape [B].
        seed (int, optional): 若给定，本次调用内 torch/numpy 随机均使用该种子（独立 RNG）；
        pairing_mode (str): ``"train"`` / ``"val"``，
            仅在同时传入 ``label`` 与 ``subject_id`` 时生效。
        train_pairing (str): ``"same_class"`` / ``"diff_class"`` / ``"both"``，
            仅在 ``pairing_mode != "val"`` 且同时传入 ``label`` 与 ``subject_id`` 时生效。

    Returns:
        torch.Tensor: Mixed output with same shape as batch_x.
    """
    if (label is None) ^ (subject_id is None):
        raise ValueError("frequency_mixing: 请同时传入 label 与 subject_id，或两者都不传。")
    if label is not None and pairing_mode not in ("train", "val"):
        raise ValueError('frequency_mixing: pairing_mode 须为 "train" 或 "val"。')
    if label is not None and train_pairing not in ("same_class", "diff_class", "both"):
        raise ValueError('frequency_mixing: train_pairing 须为 "same_class" / "diff_class" / "both"。')

    # FFT to frequency domain
    x_f = torch.fft.fft(batch_x, dim=dim)
    device = batch_x.device
    g = None
    rng_np = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        rng_np = np.random.default_rng(int(seed))

    # Random mask + dominant frequency mask (exclude DC and first harmonics)
    m = (torch.rand(x_f.shape, device=device, dtype=torch.float32, generator=g) < rate)
    amp = abs(x_f)
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)
    freal_base = x_f.real.masked_fill(m, 0)
    fimag_base = x_f.imag.masked_fill(m, 0)
    m_inv = torch.bitwise_not(m)

    def _mix_with_partner_indices(b_idx):
        x2 = batch_x[b_idx]
        x2_f = torch.fft.fft(x2, dim=dim)
        freal2 = x2_f.real.masked_fill(m_inv, 0)
        fimag2 = x2_f.imag.masked_fill(m_inv, 0)
        x_f2 = torch.complex(freal_base + freal2, fimag_base + fimag2)
        return torch.abs(torch.fft.ifft(x_f2, dim=dim))

    # Partner batch: random shuffle, or train/val structured pairing
    if label is not None:
        if train_pairing == "same_class":
            b_idx = _partner_same_class_diff_subject(label, subject_id, device, generator=g)
        elif train_pairing == "diff_class":
            b_idx = _partner_diff_class(label, subject_id, device, generator=g, prefer_diff_subject=True)
        else:
            b_idx_sc = _partner_same_class_diff_subject(label, subject_id, device, generator=g)
            b_idx_dc = _partner_diff_class(label, subject_id, device, generator=g, prefer_diff_subject=True)
            x_sc = _mix_with_partner_indices(b_idx_sc)
            x_dc = _mix_with_partner_indices(b_idx_dc)
            return torch.cat([x_sc, x_dc], dim=0)
    else:
        if rng_np is not None:
            perm = rng_np.permutation(batch_x.shape[0])
        else:
            perm = np.random.permutation(batch_x.shape[0])
        b_idx = torch.from_numpy(perm).long().to(device)
    return _mix_with_partner_indices(b_idx)


# EEG 频带 (Hz)，用于频带块替换式增强（与常见临床分段一致）
EEG_BAND_EDGES_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# 每个样本从 delta / alpha / gamma 中随机选择一个频带进行替换
_EEG_BAND_MIX_CHOICE_NAMES = ("delta", "alpha", "gamma") # , "alpha"

def _eeg_band_mix_mask_per_sample(freqs, batch_size, device, generator=None, all_bands=False):
    """
    返回 [B, n_rfft] 的 rFFT 掩码。

    all_bands=False：每个样本独立随机选一个频带，仅该频带为 True。
    all_bands=True：``_EEG_BAND_MIX_CHOICE_NAMES`` 中全部频带均为 True（并集）。

    边界与 ``EEG_BAND_EDGES_HZ`` 中各段一致（半开区间，gamma 若启用可单独处理上端点）。
    """
    masks = []
    for name in _EEG_BAND_MIX_CHOICE_NAMES:
        lo, hi = EEG_BAND_EDGES_HZ[name]
        masks.append((freqs >= lo) & (freqs < hi))
    stack = torch.stack(masks, dim=0)
    if all_bands:
        band_mask = stack.any(dim=0)
        return band_mask.unsqueeze(0).expand(batch_size, -1)
    k = stack.shape[0]
    if generator is not None:
        idx = torch.randint(0, k, (batch_size,), device=device, generator=generator)
    else:
        idx = torch.randint(0, k, (batch_size,), device=device)
    return stack[idx]


def frequency_band_mixing(
    batch_x,
    sampling_rate,
    dim=1,
    label=None,
    subject_id=None,
    seed=None,
    train_pairing="same_class",
):
    """
    按 EEG 频带块做频域替换（非随机频点）：每个样本在 rFFT 域 **随机选一个** 可混合频带
    （theta / alpha / beta，与原先多频带并集范围一致），仅将该频带系数换为 partner；
    其余频率保留 anchor。**高于所选频带之外** 的低频（含 DC、delta、未选中的 theta 等）
    与 **45 Hz 以上** 均保留 anchor，语义与原先「只替换部分频带」一致，但单次增强强度更弱。

    配对策略与 ``frequency_mixing`` 一致：若传入 ``label`` 与 ``subject_id``，由
    ``train_pairing`` 控制训练阶段配对策略；若不传入 ``label/subject_id`` 则对 batch
    随机打乱配对。

    Args:
        batch_x (torch.Tensor): 实值张量，沿 ``dim`` 为时间，形状如 [B, T, C]。
        sampling_rate (float): 采样率 (Hz)，用于将 FFT 下标映射到物理频率。
        dim (int): 时间维，默认 1（与 ``frequency_mixing`` 一致）。
        label (torch.Tensor, optional): [B] 类别。
        subject_id (torch.Tensor, optional): [B] 被试 id。
        seed (int, optional): 与 ``frequency_mixing`` 中 ``seed`` 含义相同。
        train_pairing (str): ``"same_class"`` / ``"diff_class"`` / ``"both"``，
            仅在 ``pairing_mode != "val"`` 且同时传入 ``label`` 与 ``subject_id`` 时生效。

    Returns:
        torch.Tensor: 与 ``batch_x`` 同形状的实值输出。
    """
    if (label is None) ^ (subject_id is None):
        raise ValueError("frequency_band_mixing: 请同时传入 label 与 subject_id，或两者都不传。")
    if label is not None and train_pairing not in ("same_class", "diff_class", "both"):
        raise ValueError('frequency_band_mixing: train_pairing 须为 "same_class" / "diff_class" / "both"。')

    x = batch_x.float()
    device = x.device
    n_time = x.shape[dim]
    g = None
    rng_np = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        rng_np = np.random.default_rng(int(seed))

    if label is not None:
        if train_pairing == "same_class":
            b_idx = _partner_same_class_diff_subject(label, subject_id, device, generator=g)
        elif train_pairing == "diff_class":
            b_idx = _partner_diff_class(label, subject_id, device, generator=g, prefer_diff_subject=True)
        else:
            b_idx_sc = _partner_same_class_diff_subject(label, subject_id, device, generator=g)
            b_idx_dc = _partner_diff_class(label, subject_id, device, generator=g, prefer_diff_subject=True)
            partner_sc = x[b_idx_sc]
            partner_dc = x[b_idx_dc]

            a_f = torch.fft.rfft(x, dim=dim)
            p_f_sc = torch.fft.rfft(partner_sc, dim=dim)
            p_f_dc = torch.fft.rfft(partner_dc, dim=dim)

            freqs = torch.fft.rfftfreq(n_time, d=1.0 / float(sampling_rate)).to(device)
            use_partner = _eeg_band_mix_mask_per_sample(
                freqs, x.shape[0], device, generator=g
            )

            br = [1] * a_f.ndim
            br[0] = x.shape[0]
            br[dim] = freqs.shape[0]
            m = use_partner.reshape(*br)

            mixed_f_sc = torch.where(m, p_f_sc, a_f)
            mixed_f_dc = torch.where(m, p_f_dc, a_f)
            out_sc = torch.fft.irfft(mixed_f_sc, n=n_time, dim=dim)
            out_dc = torch.fft.irfft(mixed_f_dc, n=n_time, dim=dim)
            return torch.cat([out_sc, out_dc], dim=0)
    else:
        if rng_np is not None:
            perm = rng_np.permutation(x.shape[0])
        else:
            perm = np.random.permutation(x.shape[0])
        b_idx = torch.from_numpy(perm).long().to(device)
    partner = x[b_idx]

    a_f = torch.fft.rfft(x, dim=dim)
    p_f = torch.fft.rfft(partner, dim=dim)

    freqs = torch.fft.rfftfreq(n_time, d=1.0 / float(sampling_rate)).to(device)
    use_partner = _eeg_band_mix_mask_per_sample(freqs, x.shape[0], device, generator=g, all_bands=False)  # 频带并集掩码，all_bands=True时替换所有频带

    br = [1] * a_f.ndim
    br[0] = x.shape[0]
    br[dim] = freqs.shape[0]
    m = use_partner.reshape(*br)

    mixed_f = torch.where(m, p_f, a_f)
    return torch.fft.irfft(mixed_f, n=n_time, dim=dim)


#%%
def get_augmentation(augmentation):
    if augmentation.startswith("jitter"):
        if len(augmentation) == 6:
            return Jitter()
        return Jitter(float(augmentation[6:]))
    elif augmentation.startswith("drop"):
        if len(augmentation) == 4:
            return nn.Dropout(0.1)
        return nn.Dropout(float(augmentation[4:]))
    elif augmentation.startswith("flip"):
        if len(augmentation) == 4:
            return Flip()
        return Flip(float(augmentation[4:]))
    elif augmentation.startswith("shuffle"):
        if len(augmentation) == 7:
            return Shuffle()
        return Shuffle(float(augmentation[7:]))
    elif augmentation.startswith("frequency"):
        if len(augmentation) == 9:
            return FrequencyMask()
        return FrequencyMask(float(augmentation[9:]))
    elif augmentation.startswith("mask"):
        if len(augmentation) == 4:
            return TemporalMask()
        return TemporalMask(float(augmentation[4:]))
    elif augmentation.startswith("channel"):
        if len(augmentation) == 7:
            return ChannelMask()
        return ChannelMask(float(augmentation[7:]))
    elif augmentation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown augmentation {augmentation}")



