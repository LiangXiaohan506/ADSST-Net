import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_provider.uea import (
    normalize_batch_ts,
    normalize_subject_ts,
    load_data_by_ids
)
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


def get_subject_list_adftd(args, label_path, val_ratio, test_ratio, selected_classes=None):
    '''
    Loads subject IDs for all, training, validation, and test sets for ADFTD data
    Args:
        args: arguments
        label_path: directory of label.npy file
        val_ratio: ratio of ids in validation set
        test_ratio: ratio of ids in test set
        selected_classes: list of class indices to include, e.g. [0, 1] or [0, 2]. None means all classes
    Returns:
        all_ids: list of all IDs
        train_ids: list of IDs for training set
        val_ids: list of IDs for validation set
        test_ids: list of IDs for test set
    '''
    # random shuffle to break the potential influence of human named ID order,
    # e.g., put all healthy subjects first or put subjects with more samples first, etc.
    # (which could cause data imbalance in training, validation, and test sets)
    data_list = np.load(label_path)

    # 按 selected_classes 筛选：仅保留指定类别的被试
    if selected_classes is not None:
        selected_classes = set(selected_classes)
        mask = np.isin(data_list[:, 0].astype(int), list(selected_classes))
        data_list = data_list[mask]
        if len(data_list) == 0:
            raise ValueError(f'No data found for selected_classes={selected_classes}. '
                             f'Available classes in label.npy: {np.unique(np.load(label_path)[:, 0])}')

    all_ids = list(data_list[:, 1])

    if args.cross_val == 'mccv':  # Monte Carlo cross-validation
        # 创建ID到索引的映射
        id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}

        # 第一次划分：训练集和测试集
        # train_test_split 传入两个数组时返回4个值：X_train, X_test, y_train, y_test
        train_ids, test_ids, _, _ = train_test_split(
            all_ids, data_list[:, 0], test_size=test_ratio, random_state=args.seed, stratify=data_list[:, 0]
        )

        # 获取训练集对应的索引和标签用于分层
        train_indices = [id_to_idx[id_val] for id_val in train_ids]
        train_labels = data_list[train_indices, 0]

        # 第二次划分：从训练集中分出验证集
        train_ids, val_ids, _, _ = train_test_split(
            train_ids, train_labels, test_size=val_ratio, random_state=args.seed, stratify=train_labels
        )
    else:
        raise ValueError('Invalid cross_val. Please use mccv.')

    return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)



class ADFTDLoader(Dataset):
    def __init__(self, args, flag=None, log=None):    
        self.log = log
        self.data_path = os.path.join(args.root_path, 'Feature/')
        self.label_path = os.path.join(args.root_path, 'Label/label.npy')
        root_last_seg = str(args.root_path).rstrip('/\\').split('_')[-1]
        self.augmentation_data = (root_last_seg == 'Aug')

        val_ratio, test_ratio = 0.2, 0.2
        selected_classes = getattr(args, 'selected_classes', None)
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_subject_list_adftd(
            args, self.label_path, val_ratio, test_ratio, selected_classes=selected_classes
        )
        self.selected_classes = selected_classes  # 用于后续标签重映射
        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', [int(x) for x in ids], f'(数量: {len(ids)})')
            if self.log:
                self.log.write(f'train ids: {[int(x) for x in ids]} (数量: {len(ids)})\n')
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', [int(x) for x in ids], f'(数量: {len(ids)})')
            if self.log:
                self.log.write(f'val ids: {[int(x) for x in ids]} (数量: {len(ids)})\n')
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', [int(x) for x in ids], f'(数量: {len(ids)})')
            if self.log:
                self.log.write(f'test ids: {[int(x) for x in ids]} (数量: {len(ids)})\n')
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', [int(x) for x in ids], f'(数量: {len(ids)})')
            if self.log:
                self.log.write(f'all ids: {[int(x) for x in ids]} (数量: {len(ids)})\n')
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or ALL.')

        self.X, self.y = load_data_by_ids(args, self.data_path, self.label_path, ids, self.augmentation_data) # 输出洗牌后的数据
        # 若指定了 selected_classes，将标签重映射为 0,1,...,k-1（CrossEntropyLoss 要求连续标签）
        if self.selected_classes is not None:
            class_to_idx = {c: i for i, c in enumerate(sorted(self.selected_classes))}
            self.y[:, 0] = np.array([class_to_idx[int(l)] for l in self.y[:, 0]])
        # self.X = normalize_ts(args, self.X, flag)
        self.X = normalize_batch_ts(self.X, augmentation_data=self.augmentation_data)
        # self.X = normalize_subject_ts(self.X, self.y, augmentation_data=self.augmentation_data)

        if self.augmentation_data:
            self.max_seq_len = self.X.shape[2]
        else:
            self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        # 增强布局 (2, N, T, C)：按样本维取两路视图 -> (2, T, C)；否则 (N, T, C) -> (T, C)
        if self.X.ndim == 4 and self.X.shape[0] == 2:
            x = np.ascontiguousarray(self.X[:, index, :, :])
        else:
            x = self.X[index]
        return torch.from_numpy(np.asarray(x)), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
