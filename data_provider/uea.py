import os
import numpy as np
import pandas as pd
import torch
from itertools import repeat
from scipy.signal import butter, lfilter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle



def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, y) tuples.
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor, either ``(seq_length, feat_dim)`` 或成对增强 ``(2, seq_length, feat_dim)``。
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
        max_len: global fixed sequence length. Longer sequences are clipped, shorter are padded with 0s.
    Returns:
        X: ``(batch_size, padded_length, feat_dim)`` 或 ``(batch_size, 2, padded_length, feat_dim)``（增强数据）。
        targets: (batch_size, num_labels)
        padding_masks: (batch_size, padded_length)，两路增强共用同一时间维 mask。
    """

    batch_size = len(data)
    features, labels = zip(*data)

    x0 = features[0]
    use_pair = x0.ndim == 3 and x0.shape[0] == 2

    if use_pair:
        lengths = [X.shape[1] for X in features]
        feat_dim = x0.shape[2]
    else:
        lengths = [X.shape[0] for X in features]
        feat_dim = x0.shape[-1]

    if max_len is None:
        max_len = max(lengths)

    if use_pair:
        X = torch.zeros(batch_size, 2, max_len, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :, :end, :] = features[i][:, :end, :]
    else:
        X = torch.zeros(batch_size, max_len, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)  # convert to same type as lengths tensor
        .repeat(batch_size, 1)  # (batch_size, max_len)
        .lt(lengths.unsqueeze(1))
    )


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(
        self,
        norm_type="standardization",
        mean=None,
        std=None,
        min_val=None,
        max_val=None,
    ):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


def bandpass_filter_func(signal, fs, lowcut, highcut):
    # length of signal
    fft_len = signal.shape[1]
    # FFT
    fft_spectrum = np.fft.rfft(signal, n=fft_len, axis=1)
    # get frequency bins
    freqs = np.fft.rfftfreq(fft_len, d=1/fs)
    # create mask for freqs
    mask = (freqs >= lowcut) & (freqs <= highcut)
    # expand mask to match fft_spectrum dimensions
    mask = mask[:, np.newaxis]  # Adjust mask shape if necessary
    # apply mask
    fft_spectrum = fft_spectrum * mask
    # IFFT
    filtered_signal = np.fft.irfft(fft_spectrum, n=fft_len, axis=1)

    return filtered_signal


def normalize_ts(args, ts, flag):
    """Normalize time-series data using training set statistics.

    Args:
        args: arguments object to store/retrieve train_mean_values and train_std_values.
        ts (numpy.ndarray): A batch of input time-series in shape (N, T, C).
        flag (str): "TRAIN" for training set to fit normalization parameters; otherwise use stored parameters for validation/test.

    Returns:
        numpy.ndarray: A batch of processed time-series, normalized for each channel.
    """
    if flag == "TRAIN":
        args.train_mean_values = ts.mean(axis=1, keepdims=True).mean(axis=0, keepdims=True)  # Shape: (1, 1, C)
        args.train_std_values = ts.std(axis=1, keepdims=True).mean(axis=0, keepdims=True)  # Shape: (1, 1, C)

    # Perform standard normalization
    normalized_ts = (ts - args.train_mean_values) / (args.train_std_values + 1e-8)  # Add small value to avoid division by zero

    return normalized_ts


def normalize_batch_ts(batch, augmentation_data=False):
    """Normalize a batch of time-series data (per-sample, per-channel along time).

    Args:
        batch (numpy.ndarray): Either ``(N, T, C)`` or, when ``use_augmentation=True``,
            ``(2, N, T, C)`` where维度 0 为 [原始, 增强]，两路各自独立做与单路相同的时间维标准化。
        augmentation_data (bool): ``True`` 表示 ``batch`` 为增强数据布局 ``(2, N, T, C)``。

    Returns:
        numpy.ndarray: 与输入同形状，``(N, T, C)`` 或 ``(2, N, T, C)``。
    """
    if augmentation_data:
        if batch.ndim != 4 or batch.shape[0] != 2:
            raise ValueError(
                "use_augmentation=True 时期望 batch 形状为 (2, N, T, C)，"
                f"收到 ndim={batch.ndim}, shape={batch.shape}"
            )
        mean_values = batch.mean(axis=2, keepdims=True)  # (2, N, 1, C)
        std_values = batch.std(axis=2, keepdims=True)
        return (batch - mean_values) / (std_values + 1e-8)

    mean_values = batch.mean(axis=1, keepdims=True)  # (N, 1, C)
    std_values = batch.std(axis=1, keepdims=True)
    return (batch - mean_values) / (std_values + 1e-8)


def normalize_subject_ts(batch, y, augmentation_data=False):
    """按受试者归一化：同一受试者的全部 epoch 共用该受试者、各通道的均值和标准差。

    与 ``normalize_batch_ts``（每个 epoch 沿时间维单独标准化）不同，这里把同一
    subject id 下所有样本的时间点放在一起估计 mean/std，再作用于该受试者的每一个 epoch。

    Args:
        batch (numpy.ndarray): ``(N, T, C)``，或增强布局 ``(2, N, T, C)``。
        y (numpy.ndarray): ``(N,)`` 受试者 ID，或 ``load_data_by_ids`` 返回的 ``(N, 2)``
            （第 0 列为类别，第 1 列为受试者 ID）。样本顺序须与 ``batch`` 的 N 维对齐。
        augmentation_data (bool): ``True`` 表示 ``batch`` 为 ``(2, N, T, C)``。
            两路各自独立、按同一套受试者划分做 per-subject 标准化。

    Returns:
        numpy.ndarray: 与输入同形状，``(N, T, C)`` 或 ``(2, N, T, C)``。
    """
    subject_ids = np.asarray(y)
    if subject_ids.ndim == 2:
        subject_ids = subject_ids[:, 1]
    subject_ids = np.asarray(subject_ids).reshape(-1)

    def _apply(samples):
        """samples: (N, T, C)。同一 subject 的全部时间点共用 (1, 1, C) 的 mean/std。"""
        out = np.empty_like(samples)
        for sid in np.unique(subject_ids):
            idx = np.flatnonzero(subject_ids == sid)
            sub = samples[idx]
            mean_values = sub.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
            std_values = sub.std(axis=(0, 1), keepdims=True)
            out[idx] = (sub - mean_values) / (std_values + 1e-8)
        return out

    if augmentation_data:
        if batch.ndim != 4 or batch.shape[0] != 2:
            raise ValueError(
                "use_augmentation=True 时期望 batch 形状为 (2, N, T, C)，"
                f"收到 ndim={batch.ndim}, shape={batch.shape}"
            )
        if subject_ids.shape[0] != batch.shape[1]:
            raise ValueError(
                "受试者 ID 数量须与样本数 N 一致，"
                f"收到 len(subject_ids)={subject_ids.shape[0]}, N={batch.shape[1]}"
            )
        out = np.empty_like(batch)
        for view in range(batch.shape[0]):
            out[view] = _apply(batch[view])
        return out

    if batch.ndim != 3:
        raise ValueError(
            "use_augmentation=False 时期望 batch 形状为 (N, T, C)，"
            f"收到 ndim={batch.ndim}, shape={batch.shape}"
        )
    if subject_ids.shape[0] != batch.shape[0]:
        raise ValueError(
            "受试者 ID 数量须与样本数 N 一致，"
            f"收到 len(subject_ids)={subject_ids.shape[0]}, N={batch.shape[0]}"
        )
    return _apply(batch)


def load_data_by_ids(args, data_path, label_path, ids, augmentation_data, Overlap_Ratio_Analysis=False):
    '''
    Loads subjects with IDs in the ids list
    Args:
        args: arguments（可用 ``args.use_augmentation`` 控制是否按成对增强格式加载）
        data_path: directory of data files
        label_path: directory of label.npy file
        ids: list of subject IDs to load
        augmentation_data: whether to use augmentation data
        Overlap_Ratio_Analysis: 为 True 时，按参考样本数文件对每个受试者随机下采样，
            使各受试者样本数与 USTBMCI3_1.0s_0.0_EO_Aug 数据集一致
    Returns:
        X: ``(N, T, C)`` 或当 ``use_augmentation=True`` 时为 ``(2, N, T, C)``（维度 0 为 [原始, 增强]）
        y: (N, 2), first column is label, second column is subject ID
    '''
    feature_list = []
    ori_list = []
    aug_list = []
    label_list = []
    subject_labels = np.load(label_path)

    subject_target_counts = None
    subsample_rng = None
    if Overlap_Ratio_Analysis:
        sample_counts_path = getattr(args, 'overlap_ratio_sample_counts_path', None)
        if sample_counts_path is None:
            output_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(label_path)))
            sample_counts_path = os.path.join(
                output_base_dir, "USTBMCI3_1.0s_0.0_EO_Aug_subject_sample_counts.npy"
            )
        if not os.path.exists(sample_counts_path):
            raise FileNotFoundError(
                f"Overlap_Ratio_Analysis=True 但未找到样本数文件: {sample_counts_path}"
            )
        counts_array = np.load(sample_counts_path)
        subject_target_counts = {int(row[1]): int(row[2]) for row in counts_array}
        subsample_rng = np.random.RandomState(args.seed)

    def _select_sample_indices(n_samples, sub_id):
        if not Overlap_Ratio_Analysis:
            return np.arange(n_samples)
        if sub_id not in subject_target_counts:
            raise KeyError(
                f"受试者 {sub_id} 不在样本数参考文件 {sample_counts_path} 中"
            )
        n_target = subject_target_counts[sub_id]
        if n_target == n_samples:
            return np.arange(n_samples)
        if n_target > n_samples:
            return subsample_rng.choice(n_samples, size=n_target, replace=True)
        return subsample_rng.choice(n_samples, size=n_target, replace=False)

    # load data by subject ids
    for filename in os.listdir(data_path):
        # get subject ID from filename, e.g., 'AD_1.npy'
        sub_id = int(filename.split('_')[-1].split('.')[0])
        # only load subject with ID in the ids list
        if sub_id in ids:
            # get label for the subject, all samples in the subject have the same label
            subject_label = subject_labels[subject_labels[:, 1] == sub_id][0]
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            if augmentation_data:
                sf = subject_feature
                if sf.ndim == 4 and sf.shape[0] == 2:
                    # (2, S, T, C)：按样本维 S 拆成原始/增强两路
                    for s in _select_sample_indices(sf.shape[1], sub_id):
                        ori_list.append(sf[0, s])
                        aug_list.append(sf[1, s])
                        label_list.append(subject_label)
                else:
                    raise ValueError(
                        f"use_augmentation=True 时期望每个受试者 .npy 为 (2, S, T, C)，"
                        f"收到 {filename} shape={getattr(sf, 'shape', None)}"
                    )
            else:
                sample_indices = _select_sample_indices(len(subject_feature), sub_id)
                for s in sample_indices:
                    feature_list.append(subject_feature[s])
                    label_list.append(subject_label)

    y = np.array(label_list)
    if augmentation_data:
        X = np.stack([np.array(ori_list), np.array(aug_list)], axis=0)  # (2, N, T, C)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(y))
        X = X[:, perm, :, :]
        y = y[perm]
    else:
        X = np.array(feature_list)
        X, y = shuffle(X, y, random_state=args.seed)

    return X, y



#%%
class CustomGroupSampler(torch.utils.data.Sampler):
    """ A custom Sampler sort samples by subject IDs first,
        group xxx samples into groups, shuffle the groups,
        then concatenate the groups into a list of indices,
        finally we put then into batches and shuffle the samples within each batch.

        This is used for subject-level contrastive pretraining to guarantee that there are samples from the same subject in a batch,
        avoiding all the samples from the same subjects, and maintaining the different order of samples each epoch.
    """
    def __init__(self, args, dataset, batch_size=128, group_size=2):
        super(CustomGroupSampler, self).__init__()
        self.args = args
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.indices = self.create_indices()

    def create_indices(self):
        # 固定随机种子，保证洗牌可复现
        seed = getattr(self.args, 'seed', 42)
        rng = np.random.RandomState(seed=seed)

        # 获取所有受试者ID
        subject_ids = self.dataset.y[:, 1]
        unique_subjects = np.unique(subject_ids)
        
        # 按受试者分组数据
        subject_indices_dict = {}
        for subject_id in unique_subjects:
            subject_indices_dict[subject_id] = np.where(subject_ids == subject_id)[0]
        
        # 存储所有完整的组
        all_groups = []
        
        # 处理每个受试者的数据
        for subject_id in unique_subjects:
            # 获取该受试者的所有数据索引
            subject_indices = subject_indices_dict[subject_id].copy()
            
            # 洗牌该受试者的数据（使用固定种子的 rng，可复现）
            rng.shuffle(subject_indices)
            
            # 将该受试者的数据按group_size分组
            num_groups = len(subject_indices) // self.group_size
            remainder = len(subject_indices) % self.group_size
            
            # 存储该受试者的完整组（用于后续补充）
            subject_groups = []
            
            # 提取完整的组
            for i in range(num_groups):
                group = subject_indices[i * self.group_size:(i + 1) * self.group_size]
                subject_groups.append(group)
                all_groups.append(group)
            
            # 如果有剩余数据，从该受试者的其他已分组数据中随机选取补充
            if remainder > 0:
                remaining = subject_indices[num_groups * self.group_size:]
                needed = self.group_size - remainder
                
                # 从该受试者的已分组数据中随机选取补充数据
                # 收集该受试者所有已分组的索引
                all_subject_grouped_indices = []
                for group in subject_groups:
                    all_subject_grouped_indices.extend(group)
                
                # 从已分组的索引中随机选取needed个
                if len(all_subject_grouped_indices) >= needed:
                    supplement = rng.choice(all_subject_grouped_indices, size=needed, replace=False)
                    # 组成新的完整组
                    new_group = np.concatenate([remaining, supplement])
                    all_groups.append(new_group)
                else:
                    # 如果已分组的数据不足以补充，则将所有剩余数据作为一个不完整的组（这种情况应该很少见）
                    all_groups.append(remaining)
        
        # 洗牌所有成组的数据（使用固定种子的 rng）
        rng.shuffle(all_groups)
        
        # 将所有组连接起来
        final_indices = np.concatenate(all_groups) if all_groups else np.array([], dtype=int)
        
        # 进行batch的分割
        batches = [final_indices[i:i + self.batch_size] for i in range(0, len(final_indices), self.batch_size)]
        # 洗牌每个batch内的样本（使用固定种子的 rng）
        for batch in batches:
            rng.shuffle(batch)
        # 连接所有batch
        final_indices = np.concatenate(batches) if batches else np.array([], dtype=int)
        
        return final_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.dataset)