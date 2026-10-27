import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.signal import hilbert
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.preprocessing import label_binarize
plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_model = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.save_model = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.save_model = False
            print(f"EarlyStopping counter: {self.counter} out of {self.patience} \n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.save_model = True

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Metric score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n"
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


#%%
def calculate_subject_level_metrics(predictions, true_labels, subject_ids, num_classes,
                                    pred_probs=None, trim_ratio=0.0):
    """
    pred_probs: (N, num_classes) 每个样本的预测概率，用于按置信度做 trimmed mean。若为 None 则不裁剪。
    trim_ratio: 丢弃置信度最低的比例，例如 0.2 表示丢掉最低 20% 的窗再投票；0 表示不丢（等价于原多数投票）。
    """
    # Step 1: Get unique subject_ids
    unique_subjects = np.unique(subject_ids)

    # Step 2: Aggregate predictions and true labels for each subject_id
    subject_predictions = []
    subject_trues = []

    for subject in unique_subjects:
        # Find all sample indices for the current subject
        indices = np.where(subject_ids == subject)[0]

        # Get predictions and true labels for the current subject
        subject_preds = predictions[indices]
        subject_true = true_labels[indices][0]  # The true_label should be the same for all samples of a subject

        # Trimmed mean: 按置信度丢弃最低 trim_ratio 比例的窗，再对剩余窗做多数投票
        if pred_probs is not None and trim_ratio > 0 and len(indices) > 0:
            subject_probs = pred_probs[indices]  # (K, num_classes)
            confidence = np.max(subject_probs, axis=1)  # (K,) 每个窗的预测置信度
            n_trim = max(0, int(np.ceil(len(indices) * trim_ratio)))
            if n_trim < len(indices):
                # 保留置信度最高的 (1 - trim_ratio) 的窗
                order = np.argsort(confidence)
                keep_order = order[n_trim:]
                subject_preds = subject_preds[keep_order]

        # Determine the majority vote prediction for the subject (可能只剩 1 个样本)
        if len(subject_preds) == 0:
            subject_preds = predictions[indices]  # 回退：全部参与投票
        majority_label = Counter(subject_preds).most_common(1)[0][0]
        # avg_prob = np.mean(subject_preds, axis=0)
        # majority_label = np.argmax(avg_prob)

        # Record subject-level results
        subject_predictions.append(majority_label)
        subject_trues.append(subject_true)

    # Convert to numpy arrays
    subject_predictions = np.array(subject_predictions)
    subject_trues = np.array(subject_trues)

    # Step 3: Calculate metrics
    # Convert true labels to one-hot encoding for AUROC and AUPRC
    subject_true_onehot = label_binarize(subject_trues, classes=list(range(num_classes)))
    subject_probs = label_binarize(subject_predictions, classes=list(range(num_classes)))  # Simplified assumption

    metrics = {
        "Accuracy": accuracy_score(subject_trues, subject_predictions),
        "Precision": precision_score(subject_trues, subject_predictions, average="macro"),
        "Recall": recall_score(subject_trues, subject_predictions, average="macro"),
        "F1": f1_score(subject_trues, subject_predictions, average="macro"),
        "k-score": cohen_kappa_score(subject_trues, subject_predictions),
        "AUROC": roc_auc_score(subject_true_onehot, subject_probs, multi_class="ovr"),
        "AUPRC": average_precision_score(subject_true_onehot, subject_probs, average="macro"),
        "Confusion Matrix": confusion_matrix(subject_trues, subject_predictions),
    }

    return metrics


#%%
def _get_loss_scalar(loss_item):
    """支持 test_loss 为 float 或 dict {'total', 'ce', 'cons'}"""
    return loss_item['total'] if isinstance(loss_item, dict) else loss_item


def compute_avg_std(args, sample_test_metrics_dict_list, subject_test_metrics_dict_list):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>> 所有ITR的平均结果 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # 保存结果到文件
    folder_path = ("./results/" + args.project_name + "/")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = "results.txt"
    file_path = os.path.join(folder_path, file_name)

    # 写入每个ITR的详细结果
    with open(file_path, 'a') as f:
        f.write("=" * 80 + "\n")
        f.write(f"实验配置: {args.project_name}\n")
        f.write("=" * 80 + "\n\n")

        # 写入每个ITR的结果（支持 test_loss 为 float 或 dict）
        for itr in range(len(sample_test_metrics_dict_list)):
            f.write(f"ITR {itr + 1} 结果:\n")

            # Sample-level results
            f.write("\tSample-level results:\n")
            sample_metrics = sample_test_metrics_dict_list[itr]
            for key, value in sample_metrics.items():
                if key != "Confusion Matrix":
                    f.write(f"\t\t{key}:\t{value * 100:.2f}%\n")
                else:
                    # 保存原始混淆矩阵
                    f.write(f"\t\t{key} (Count):\n")
                    for row in value:
                        f.write("\t\t\t" + "\t".join([str(int(x)) for x in row]) + "\n")
                    # 计算并保存百分比版本的混淆矩阵
                    cm_percent = value.astype(float)
                    row_sums = cm_percent.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1  # 避免除零
                    cm_percent = (cm_percent / row_sums) * 100
                    f.write(f"\t\t{key} (Percentage):\n")
                    for row in cm_percent:
                        f.write("\t\t\t" + "\t".join([f"{x:.2f}%" for x in row]) + "\n")

            # Subject-level results
            f.write("\tSubject-level results:\n")
            subject_metrics = subject_test_metrics_dict_list[itr]
            for key, value in subject_metrics.items():
                if key != "Confusion Matrix":
                    f.write(f"\t\t{key}:\t{value * 100:.2f}%\n")
                else:
                    # 保存原始混淆矩阵
                    f.write(f"\t\t{key} (Count):\n")
                    for row in value:
                        f.write("\t\t\t" + "\t".join([str(int(x)) for x in row]) + "\n")
                    # 计算并保存百分比版本的混淆矩阵
                    cm_percent = value.astype(float)
                    row_sums = cm_percent.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1  # 避免除零
                    cm_percent = (cm_percent / row_sums) * 100
                    f.write(f"\t\t{key} (Percentage):\n")
                    for row in cm_percent:
                        f.write("\t\t\t" + "\t".join([f"{x:.2f}%" for x in row]) + "\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("所有ITR的平均结果 (Mean ± Std):\n")
        f.write("=" * 80 + "\n\n")
    
    # 计算平均值和标准差（支持 test_loss 为 float 或 dict）
    sample_test_metrics_dict_avg_std = {}
    subject_test_metrics_dict_avg_std = {}
    
    # Compute average and std
    for key in sample_test_metrics_dict_list[0].keys():
        if key != "Confusion Matrix":
            # convert to percentage
            sample_test_avg = np.mean([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100
            sample_test_std = np.std([test_metrics_dict[key] for test_metrics_dict in sample_test_metrics_dict_list]) * 100
            subject_test_avg = np.mean([test_metrics_dict[key] for test_metrics_dict in subject_test_metrics_dict_list]) * 100
            subject_test_std = np.std([test_metrics_dict[key] for test_metrics_dict in subject_test_metrics_dict_list]) * 100

            sample_test_metrics_dict_avg_std[key] = (sample_test_avg, sample_test_std)
            subject_test_metrics_dict_avg_std[key] = (subject_test_avg, subject_test_std)
    
    # 收集混淆矩阵并计算平均值和标准差
    if "Confusion Matrix" in sample_test_metrics_dict_list[0]:
        # 收集所有混淆矩阵
        sample_cm_list = [test_metrics_dict["Confusion Matrix"] for test_metrics_dict in sample_test_metrics_dict_list]
        subject_cm_list = [test_metrics_dict["Confusion Matrix"] for test_metrics_dict in subject_test_metrics_dict_list]
        
        # 转换为numpy数组并计算平均值和标准差
        sample_cm_array = np.array(sample_cm_list)
        subject_cm_array = np.array(subject_cm_list)
        
        sample_cm_mean = np.mean(sample_cm_array, axis=0)
        sample_cm_std = np.std(sample_cm_array, axis=0)
        subject_cm_mean = np.mean(subject_cm_array, axis=0)
        subject_cm_std = np.std(subject_cm_array, axis=0)
        
        sample_test_metrics_dict_avg_std["Confusion Matrix"] = (sample_cm_mean, sample_cm_std)
        subject_test_metrics_dict_avg_std["Confusion Matrix"] = (subject_cm_mean, subject_cm_std)

    # Format results into a single line with two decimal places and percentages
    sample_test_results = "Test results --- " + ", ".join(
        [f"{key}: {sample_test_metrics_dict_avg_std[key][0]:.2f} ± {sample_test_metrics_dict_avg_std[key][1]:.2f}%"
         for key in sample_test_metrics_dict_avg_std.keys() if key != "Confusion Matrix"]
    )
    subject_test_results = "Test results --- " + ", ".join(
        [f"{key}: {subject_test_metrics_dict_avg_std[key][0]:.2f} ± {subject_test_metrics_dict_avg_std[key][1]:.2f}%"
         for key in subject_test_metrics_dict_avg_std.keys() if key != "Confusion Matrix"]
    )

    # 写入平均结果到文件
    with open(file_path, 'a') as f:
        f.write('Sample-level results:\n')
        f.write(sample_test_results + "\n")
        # 写入混淆矩阵的平均值和标准差
        if "Confusion Matrix" in sample_test_metrics_dict_avg_std:
            cm_mean, cm_std = sample_test_metrics_dict_avg_std["Confusion Matrix"]
            f.write(f"\tConfusion Matrix (Mean Count):\n")
            for row in cm_mean:
                f.write("\t\t" + "\t".join([f"{x:.2f}" for x in row]) + "\n")
            f.write(f"\tConfusion Matrix (Std Count):\n")
            for row in cm_std:
                f.write("\t\t" + "\t".join([f"{x:.2f}" for x in row]) + "\n")
            # 计算百分比版本
            cm_mean_percent = cm_mean.astype(float)
            row_sums = cm_mean_percent.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            cm_mean_percent = (cm_mean_percent / row_sums) * 100
            f.write(f"\tConfusion Matrix (Mean Percentage):\n")
            for row in cm_mean_percent:
                f.write("\t\t" + "\t".join([f"{x:.2f}%" for x in row]) + "\n")
        f.write("\n")
        f.write('Subject-level results after majority voting:\n')
        f.write(subject_test_results + "\n")
        # 写入混淆矩阵的平均值和标准差
        if "Confusion Matrix" in subject_test_metrics_dict_avg_std:
            cm_mean, cm_std = subject_test_metrics_dict_avg_std["Confusion Matrix"]
            f.write(f"\tConfusion Matrix (Mean Count):\n")
            for row in cm_mean:
                f.write("\t\t" + "\t".join([f"{x:.2f}" for x in row]) + "\n")
            f.write(f"\tConfusion Matrix (Std Count):\n")
            for row in cm_std:
                f.write("\t\t" + "\t".join([f"{x:.2f}" for x in row]) + "\n")
            # 计算百分比版本
            cm_mean_percent = cm_mean.astype(float)
            row_sums = cm_mean_percent.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            cm_mean_percent = (cm_mean_percent / row_sums) * 100
            f.write(f"\tConfusion Matrix (Mean Percentage):\n")
            for row in cm_mean_percent:
                f.write("\t\t" + "\t".join([f"{x:.2f}%" for x in row]) + "\n")
        f.write("\n" + "=" * 80 + "\n\n")

    # 打印平均结果（无论是否训练模式都打印）
    print("\n" + "=" * 80)
    print("所有ITR的平均结果 (Mean ± Std):")
    print("=" * 80)
    print('Sample-level results:')
    print(sample_test_results)
    # 打印混淆矩阵
    if "Confusion Matrix" in sample_test_metrics_dict_avg_std:
        cm_mean, cm_std = sample_test_metrics_dict_avg_std["Confusion Matrix"]
        print("\nConfusion Matrix (Mean Count):")
        for row in cm_mean:
            print("\t" + "\t".join([f"{x:.2f}" for x in row]))
        print("Confusion Matrix (Std Count):")
        for row in cm_std:
            print("\t" + "\t".join([f"{x:.2f}" for x in row]))
        # 计算百分比版本
        cm_mean_percent = cm_mean.astype(float)
        row_sums = cm_mean_percent.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        cm_mean_percent = (cm_mean_percent / row_sums) * 100
        print("Confusion Matrix (Mean Percentage):")
        for row in cm_mean_percent:
            print("\t" + "\t".join([f"{x:.2f}%" for x in row]))
    print('\nSubject-level results after majority voting:')
    print(subject_test_results)
    # 打印混淆矩阵
    if "Confusion Matrix" in subject_test_metrics_dict_avg_std:
        cm_mean, cm_std = subject_test_metrics_dict_avg_std["Confusion Matrix"]
        print("\nConfusion Matrix (Mean Count):")
        for row in cm_mean:
            print("\t" + "\t".join([f"{x:.2f}" for x in row]))
        print("Confusion Matrix (Std Count):")
        for row in cm_std:
            print("\t" + "\t".join([f"{x:.2f}" for x in row]))
        # 计算百分比版本
        cm_mean_percent = cm_mean.astype(float)
        row_sums = cm_mean_percent.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        cm_mean_percent = (cm_mean_percent / row_sums) * 100
        print("Confusion Matrix (Mean Percentage):")
        for row in cm_mean_percent:
            print("\t" + "\t".join([f"{x:.2f}%" for x in row]))
    print("=" * 80 + "\n")



#%% GCN计算初始化图结构
channel_orders = {
    'ADFTD': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4',
              'T6', 'O1', 'O2'],
    'CAUEEG': ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F7', 'T3', 'T5', 'F8', 'T4', 'T6',
               'Fz', 'Cz', 'Pz'],
    'USTBMCI': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T7', 'T8', 'Pz', 'P3', 'P4',
                'P7', 'P8', 'Oz', 'O1', 'O2']
}

# Define brain regions (normalize channel names to handle case variations and T3/T7 equivalents)
brain_regions = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
    'Central': ['Cz', 'C3', 'C4'],
    'Parietal': ['Pz', 'P3', 'P4'],
    'Occipital': ['O1', 'O2', 'Oz'],
    'Temporal': ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'P7', 'P8']
}

def _ensure_ct(x, enc_in=None):
    """
    确保返回 (n_windows, C, T)。
    当 enc_in 提供时，根据通道数判断格式；(N,C,T) 中 C=shape[1]，(N,T,C) 中 C=shape[2]。
    当 enc_in 未提供时，使用经验判断 T 通常 > C。
    """
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got {x.shape}")
    if enc_in is not None:
        # 根据通道数辅助判断
        if x.shape[1] == enc_in and x.shape[2] != enc_in:
            return x  # (N,C,T)
        if x.shape[2] == enc_in and x.shape[1] != enc_in:
            return np.transpose(x, (0, 2, 1))  # (N,T,C) -> (N,C,T)
        # 两维均等于或均不等于时，退化为经验判断
    if x.shape[1] < x.shape[2]:  # 经验判断：T 通常 > C
        return x  # 可能已是 (N,C,T)
    return np.transpose(x, (0, 2, 1))  # (N,T,C) -> (N,C,T)

def _row_topk_mask(A, k):
    """每行保留 top-k（不含对角），返回稀疏化后的 A"""
    N = A.shape[0]
    if k is None or k <= 0 or k >= N:
        return A
    A = A.copy()
    # 临时把对角设为 -inf，避免选中自己
    np.fill_diagonal(A, -np.inf)
    idx = np.argpartition(A, -k, axis=1)[:, -k:]  # (N,k) 无序 topk
    mask = np.zeros_like(A, dtype=bool)
    rows = np.arange(N)[:, None]
    mask[rows, idx] = True
    A[~mask] = 0.0
    # 对角归零
    np.fill_diagonal(A, 0.0)
    return A

def _symmetrize(A, mode="max"):
    """
    mode:
      - "avg": (A + A.T)/2
      - "max": max(A, A.T)   (更利于保留边)
    """
    if mode == "avg":
        return 0.5 * (A + A.T)
    elif mode == "max":
        return np.maximum(A, A.T)
    else:
        raise ValueError("mode must be 'avg' or 'max'")


def _frontal_occipital_indices(dataset_name, num_channels):
    """
    根据 channel_orders 将 Frontal / Occipital 通道映射到矩阵下标。
    若 dataset_name 未知或通道数与定义不一致则返回 None。
    """
    if dataset_name is None or dataset_name not in channel_orders:
        return None
    order = channel_orders[dataset_name]
    if len(order) != num_channels:
        return None
    frontal = frozenset(brain_regions["Frontal"])
    occipital = frozenset(brain_regions["Occipital"])
    f_idx = [i for i, ch in enumerate(order) if ch in frontal]
    o_idx = [i for i, ch in enumerate(order) if ch in occipital]
    return f_idx, o_idx


def _restore_frontal_occipital_edges(A_sparse, A_dense, f_idx, o_idx):
    """将稀疏前 A_dense 中 Frontal–Occipital 跨区边写回 A_sparse（含双向，对角保持后续步骤处理）。"""
    for i in f_idx:
        for j in o_idx:
            if i == j:
                continue
            v = A_dense[i, j]
            A_sparse[i, j] = v
            A_sparse[j, i] = v
    return A_sparse

def compute_pearson_A0_sample_mean(
    data_set,                  # 数据集，具有 .X 属性，类似 exp_supervised 中的 train_data
    enc_in=None,               # 通道数，用于 _ensure_ct 辅助判断 (N,T,C) vs (N,C,T)
    use_abs=True,             # abs Pearson
    topk=6,                   # 每行保留 k 条边
    sym_mode="max",           # "max" 比 "avg" 更保边
    eps=1e-8,
    row_normalize=True,       # 让每行和为 1（适配 X @ A 传播）
    clip_negative=True        # 保证非负
):
    """
    计算所有训练集样本的平均 Pearson 相关矩阵。

    返回:
      A0: (C,C) float32
    """
    X = _ensure_ct(np.asarray(data_set.X), enc_in=enc_in)
    N, C, T = X.shape

    Rs = []
    for i in range(N):
        # 每个样本: (C, T)
        xs = X[i]  # (C, T)

        # 去均值（按通道）
        xs = xs - xs.mean(axis=1, keepdims=True)

        # 计算协方差/相关
        # cov = X X^T / (L-1)
        L = xs.shape[1]
        cov = (xs @ xs.T) / max(L - 1, 1)

        std = np.sqrt(np.diag(cov)).reshape(C, 1)
        denom = (std @ std.T) + eps
        R = cov / denom  # Pearson (C,C)

        # 数值清理
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        if use_abs:
            R = np.abs(R)
        elif clip_negative:
            R = np.maximum(R, 0.0)

        np.fill_diagonal(R, 0.0)
        Rs.append(R)

    if len(Rs) == 0:
        raise RuntimeError("data_set 无有效样本，请检查 data_set.X。")

    # 按样本平均（所有训练集样本的 Pearson 矩阵取平均）
    A0 = np.mean(np.stack(Rs, axis=0), axis=0)  # (C,C)

    # top-k 稀疏（先在有向上做）
    A0 = _row_topk_mask(A0, k=topk)

    # 对称化（得到无向图）
    A0 = _symmetrize(A0, mode=sym_mode)

    # 再次对角归零
    np.fill_diagonal(A0, 0.0)

    # 非负保障
    if clip_negative:
        A0 = np.maximum(A0, 0.0)

    # 行归一化（可选）
    if row_normalize:
        row_sum = A0.sum(axis=1, keepdims=True) + eps
        A0 = A0 / row_sum

    return A0.astype(np.float32)


def compute_pearson_A0_subject_mean(
    data_set,                  # 数据集，具有 .X 和 .y 属性，类似 exp_supervised 中的 train_data
    enc_in=None,               # 通道数，用于 _ensure_ct 辅助判断 (N,T,C) vs (N,C,T)
    use_abs=True,             # abs Pearson
    topk=6,                   # 每行保留 k 条边（20通道常用 4~8）
    sym_mode="max",           # "max" 比 "avg" 更保边
    eps=1e-8,
    row_normalize=True,       # 让每行和为 1（适配你那套 X @ A 传播）
    clip_negative=True,        # 保证非负
    dataset_name=None,
    preserve_frontal_occipital=False,
):
    """
    计算按受试者平均的 Pearson 相关矩阵。

    从 data_set 获取数据（类似 exp_supervised 中的 train_data），
    data_set.X 为 (N, T, C) 或 (N, C, T)，data_set.y[:, 1] 为 subject_ids。
    若 ``use_augmentation=True``，则 ``data_set.X`` 为 ``(2, N, T, C)``（原始与增强），
    仅取 ``X[0]`` 即 ``(N, T, C)`` 做后续 Pearson 计算。

    在 top-k 稀疏之后，若 ``preserve_frontal_occipital=True`` 且 ``dataset_name`` 在
    ``channel_orders`` 中且通道数一致，则把稀疏前的 Frontal–Occipital 跨区连接
    从稠密矩阵写回（与 ``brain_regions`` 一致）。

    返回:
      A0: (C,C) float32
    """
    raw = np.asarray(data_set.X)
    if raw.ndim == 4 and raw.shape[0] == 2:
        raw = raw[0]
    else:
        raise ValueError(
            "use_augmentation=True 时期望 data_set.X 形状为 (2, N, T, C)，"
            f"收到 ndim={raw.ndim}, shape={raw.shape}"
        )

    X = _ensure_ct(raw, enc_in=enc_in)
    subject_ids = np.asarray(data_set.y[:, 1])
    N, C, T = X.shape

    uniq_subj = np.unique(subject_ids)
    Rs = []

    for s in uniq_subj:
        xs = X[subject_ids == s]          # (Ns,C,T)
        if xs.shape[0] == 0:
            continue

        # 拼接该被试所有窗口：把时间维拉长
        # (Ns,C,T) -> (C, Ns*T)
        Xcat = xs.transpose(1, 0, 2).reshape(C, -1)

        # 去均值（按通道）
        Xcat = Xcat - Xcat.mean(axis=1, keepdims=True)

        # 计算协方差/相关
        # cov = X X^T / (L-1)
        L = Xcat.shape[1]
        cov = (Xcat @ Xcat.T) / max(L - 1, 1)

        std = np.sqrt(np.diag(cov)).reshape(C, 1)
        denom = (std @ std.T) + eps
        R = cov / denom  # Pearson (C,C)

        # 数值清理
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        if use_abs:
            R = np.abs(R)
        elif clip_negative:
            R = np.maximum(R, 0.0)

        np.fill_diagonal(R, 0.0)
        Rs.append(R)

    if len(Rs) == 0:
        raise RuntimeError("data_set 无有效受试者数据，请检查 data_set.X 与 data_set.y[:, 1]。")

    # 按被试平均（每个被试权重相等）
    A0 = np.mean(np.stack(Rs, axis=0), axis=0)  # (C,C)
    A0_dense = A0.copy()

    # top-k 稀疏（先在有向上做）
    A0 = _row_topk_mask(A0, k=topk)

    if preserve_frontal_occipital:
        fo = _frontal_occipital_indices(dataset_name, C)
        if fo is not None:
            f_idx, o_idx = fo
            _restore_frontal_occipital_edges(A0, A0_dense, f_idx, o_idx)

    # 对称化（得到无向图）
    A0 = _symmetrize(A0, mode=sym_mode)

    # 再次对角归零
    np.fill_diagonal(A0, 0.0)

    # 非负保障
    if clip_negative:
        A0 = np.maximum(A0, 0.0)

    # 行归一化（可选）
    if row_normalize:
        row_sum = A0.sum(axis=1, keepdims=True) + eps
        A0 = A0 / row_sum

    return A0.astype(np.float32)


_PLI_WINDOW_TAG = re.compile(r"_\d+(?:\.\d+)?s_")
_PLI_CONTINUOUS_SUFFIX = "_180.0s_0.0_EO"


def _pli_continuous_feature_dir(dataset_folder):
    """
    从训练数据集文件夹名截取前缀，拼出连续 180s EO 的 Feature 目录。

    ``CAUEEG_Dementia_1.0s_0.0_EO_Aug`` → ``CAUEEG_Dementia_180.0s_0.0_EO``
    ``USTBMCI3_1.0s_0.0_EO_Aug`` → ``USTBMCI3_180.0s_0.0_EO``
    也可传入 ``./dataset/<文件夹名>`` 这样的 root_path。
    """
    name = os.path.basename(os.path.normpath(str(dataset_folder)))
    match = _PLI_WINDOW_TAG.search(name)
    if match is None or match.start() == 0:
        raise ValueError(
            "无法从数据集名称中截取前缀，期望形如 "
            f"'USTBMCI3_1.0s_0.0_EO_Aug'，收到 {dataset_folder!r}"
        )
    folder = name[: match.start()] + _PLI_CONTINUOUS_SUFFIX
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        folder,
        "Feature",
    )

def _pli_matrix_from_continuous(x, time_chunk=256):
    """
    由一名受试者切窗前的连续 EEG 计算 PLI（Phase Lag Index, Stam et al., 2007）。

    x: (T, C) 或 (1, T, C)，例如 (23040, 20) / (1, 23040, 20)。
    整段只做一次 Hilbert，不在窗口边界处重新估计相位。
    PLI_ij = |mean_t sign(Im(z_i(t) conj(z_j(t))))|。

    返回 (C, C)，取值 [0, 1]，对角为 0。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"连续 EEG 期望 (1, T, C)，收到 {x.shape}")
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"连续 EEG 期望 (T, C) 或 (1, T, C)，收到 {x.shape}")

    # (T, C) -> (C, T)
    x = np.ascontiguousarray(x.T)
    C, T = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    z = hilbert(x, axis=-1)
    real = np.real(z)
    imag = np.imag(z)

    acc = np.zeros((C, C), dtype=np.float64)
    for t0 in range(0, T, time_chunk):
        t1 = min(T, t0 + time_chunk)
        r = real[:, t0:t1]
        im = imag[:, t0:t1]
        # Im(z_i * conj(z_j)) = im_i * r_j - r_i * im_j
        cross_im = im[:, None, :] * r[None, :, :] - r[:, None, :] * im[None, :, :]
        acc += np.sign(cross_im).sum(axis=-1)

    pli = np.abs(acc / max(T, 1))
    pli = np.nan_to_num(pli, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(pli, 0.0)
    return pli


def _pli_feature_subject_id(filename):
    """``feature_00587.npy`` / ``feature_01.npy`` → 受试者 ID。"""
    return int(os.path.splitext(filename)[0].split("_")[-1])


def compute_pli_A0_subject_mean(
    dataset_folder,           # 训练数据文件夹名或 root_path，如 USTBMCI3_1.0s_0.0_EO_Aug
    subject_ids,              # 仅使用这些受试者，通常为当前折的 train_ids
    topk=6,                   # 每行保留 k 条边（20通道常用 4~8）
    sym_mode="max",           # "max" 比 "avg" 更保边
    eps=1e-8,
    row_normalize=True,       # 让每行和为 1（适配 X @ A 传播）
    clip_negative=True,        # 保证非负
    dataset_name="USTBMCI",
    preserve_frontal_occipital=False,
):
    """
    计算按受试者平均的 PLI（Phase Lag Index）矩阵。

    不接收已切窗的训练集。``dataset_folder`` 为当前训练数据的文件夹名
    （或 ``./dataset/<文件夹名>``）。截取窗口标记（如 ``_1.0s_``）之前的前缀，
    再拼上 ``_180.0s_0.0_EO``，从 ``dataset/<前缀>_180.0s_0.0_EO/Feature``
    读取连续睁眼 EEG。每个文件形状 ``(1, T, C)``，例如 ``(1, 23040, 20)``。

    ``subject_ids`` 指定参与平均的受试者（当前折训练集）。文件名末尾数字与
    ``label.npy`` 中的受试者 ID 对齐，验证集和测试集文件不参与计算。

    每个受试者在整段连续信号上做一次 Hilbert，按
    PLI_ij = |mean_t sign(Im(z_i(t) conj(z_j(t))))| 得到一张 (C, C) 矩阵
    （零滞后不同步，取值 [0, 1]）。各受试者等权平均。

    在 top-k 稀疏之后，若 ``preserve_frontal_occipital=True`` 且 ``dataset_name`` 在
    ``channel_orders`` 中且通道数一致，则把稀疏前的 Frontal–Occipital 跨区连接
    从稠密矩阵写回（与 ``brain_regions`` 一致）。

    返回:
      A0: (C,C) float32
    """
    wanted = {int(subject_id) for subject_id in subject_ids}
    if not wanted:
        raise ValueError("subject_ids 为空，无法计算训练集 PLI")

    feature_dir = _pli_continuous_feature_dir(dataset_folder)
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"未找到连续 EO 特征目录: {feature_dir}")

    files_by_id = {}
    for name in os.listdir(feature_dir):
        if not (name.startswith("feature_") and name.endswith(".npy")):
            continue
        subject_id = _pli_feature_subject_id(name)
        if subject_id not in wanted:
            continue
        if subject_id in files_by_id:
            raise RuntimeError(
                f"{feature_dir} 中受试者 {subject_id} 有多个特征文件: "
                f"{files_by_id[subject_id]}, {name}"
            )
        files_by_id[subject_id] = name

    missing = sorted(wanted - set(files_by_id))
    if missing:
        preview = missing[:20]
        extra = "" if len(missing) <= 20 else f" 等共 {len(missing)} 人"
        raise FileNotFoundError(
            f"{feature_dir} 缺少训练集受试者的连续特征: {preview}{extra}"
        )

    print(
        f"PLI A0: 使用训练集受试者 {len(wanted)} 人"
        f"（目录 {feature_dir}）"
    )

    Rs = []
    for subject_id in sorted(files_by_id):
        x = np.load(os.path.join(feature_dir, files_by_id[subject_id]))
        R = _pli_matrix_from_continuous(x)
        Rs.append(R)

    # 按被试平均（每个被试权重相等）
    A0 = np.mean(np.stack(Rs, axis=0), axis=0)  # (C,C)
    C = A0.shape[0]
    A0_dense = A0.copy()

    # top-k 稀疏（先在有向上做）
    A0 = _row_topk_mask(A0, k=topk)

    if preserve_frontal_occipital:
        fo = _frontal_occipital_indices(dataset_name, C)
        if fo is not None:
            f_idx, o_idx = fo
            _restore_frontal_occipital_edges(A0, A0_dense, f_idx, o_idx)

    # 对称化（得到无向图）
    A0 = _symmetrize(A0, mode=sym_mode)

    # 再次对角归零
    np.fill_diagonal(A0, 0.0)

    # 非负保障
    if clip_negative:
        A0 = np.maximum(A0, 0.0)

    # 行归一化（可选）
    if row_normalize:
        row_sum = A0.sum(axis=1, keepdims=True) + eps
        A0 = A0 / row_sum

    return A0.astype(np.float32)


def generate_random_A0(
    enc_in,
    topk=6,
    sym_mode="max",
    eps=1e-8,
    row_normalize=True,
    clip_negative=True,
    dataset_name=None,
    preserve_frontal_occipital=False,
    seed=None,
):
    """
    随机生成 A0 邻接矩阵（不依赖训练数据），数值均匀采样于 [0, 1]。

    后处理流程与 ``compute_pearson_A0_subject_mean`` 一致：
    对角归零 → top-k 行稀疏 →（可选）恢复 Frontal–Occipital 边 → 对称化 →
    非负裁剪 → 行归一化。

    参数:
      enc_in: 通道数 C，输出形状为 (C, C)。
      seed: 随机种子；为 None 时不固定。

    返回:
      A0: (C, C) float32
    """
    if enc_in is None or enc_in <= 0:
        raise ValueError(f"enc_in 须为正整数，收到 enc_in={enc_in!r}")

    C = int(enc_in)
    rng = np.random.default_rng(seed)

    A0 = rng.uniform(0.0, 1.0, size=(C, C))
    np.fill_diagonal(A0, 0.0)
    A0_dense = A0.copy()

    A0 = _row_topk_mask(A0, k=topk)

    if preserve_frontal_occipital:
        fo = _frontal_occipital_indices(dataset_name, C)
        if fo is not None:
            f_idx, o_idx = fo
            _restore_frontal_occipital_edges(A0, A0_dense, f_idx, o_idx)

    A0 = _symmetrize(A0, mode=sym_mode)
    np.fill_diagonal(A0, 0.0)

    if clip_negative:
        A0 = np.maximum(A0, 0.0)

    if row_normalize:
        row_sum = A0.sum(axis=1, keepdims=True) + eps
        A0 = A0 / row_sum

    return A0.astype(np.float32)


