# ADSST-Net

ADSST-Net 是一个面向脑电（EEG）分类的研究工程，包含模型实现、数据预处理、监督训练、样本级与受试者级评估，以及特征可解释性分析。当前工程支持 ADFTD、CAEEG、CAUEEG 和 IUMSMCI 四类数据加载器。

## 框架概览

![Overview of the Proposed Framework](figs/Overview%20of%20the%20Proposed%20Framework.png)

模型主要由以下模块构成：

- **自适应频谱调制（ASM）**：对频谱与通道信息进行自适应调制。
- **双向状态空间建模（BiSTS）**：通过 Mamba 相关结构建模序列信息。
- **先验引导图学习（PGL）**：以训练受试者的相关性构建图先验，并学习通道间关系。
- **门控差分交互与融合（RGDI）**：对序列分支与图分支进行门控交互、差分抑制和特征融合。
- **多层投影与分类**：结合分类损失及可配置的实例级、受试者级对比学习与正则项。

实现以 `models/ADSSTNet.py` 和 `layers/ADSSTNet_EncDec.py` 为准。本仓库不声明未经验证的性能结果。

## 目录结构

```text
ADSST-Net/
├── run.py                              # 实验入口与参数配置
├── models/                             # ADSSTNet 及对比模型
├── layers/                             # ASM、PGL、BiSTS、交互融合等模块
├── exp/                                # 训练、验证与测试逻辑
├── data_provider/                      # 数据加载、采样与批处理
├── data_preprocessing/                 # 数据预处理及划分 notebooks
├── scripts/                            # 各数据集实验与消融配置
├── Feature and Visualization Analysis/ # 特征分析与可解释性 notebooks
├── utils/                              # 损失函数、评估及辅助工具
├── figs/                               # 框架图
├── requirements.txt                    # 原工程依赖版本记录
└── LICENSE
```

数据集、训练权重、运行结果及本地 IDE 缓存不随代码上传，运行时需自行准备数据。

## 环境与安装

原工程记录的 Python 版本为 **3.10.14**，PyTorch 版本为 **2.5.1 + CUDA 11.8**。由于模型依赖 Mamba、Triton 相关算子，建议使用 Linux 与 NVIDIA CUDA 环境；当前未验证 macOS 或纯 CPU 运行。

```bash
git clone https://github.com/LiangXiaohan506/ADSST-Net.git
cd ADSST-Net
conda create -n adsstnet python=3.10.14 -y
conda activate adsstnet

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# requirements.txt 中的 python 条目用于记录解释器版本；PyTorch 已在上一步安装。
python -c "from pathlib import Path; p=Path('requirements.txt'); Path('requirements-runtime.txt').write_text('\n'.join(s for s in p.read_text().splitlines() if not s.strip().startswith(('python==', 'torch')))+'\n')"
pip install -r requirements-runtime.txt

# 下列包在源码中被导入，但原 requirements.txt 尚未列出。
pip install timm wandb visdom
pip install causal-conv1d mamba-ssm --no-build-isolation
```

Mamba 及其 CUDA 扩展需与本机 PyTorch、CUDA 和编译工具链匹配；以上补充依赖没有锁定版本，完整环境尚未在干净环境中验证。预处理与分析 notebooks 另需 Jupyter；模型 FLOPs/MACs 统计可选安装 `thop`。

## 数据准备

使用 `data_preprocessing/` 中对应的 notebook 准备数据，并根据本机路径修改 notebook 内的数据位置。各数据集需自行获取并遵守其使用要求。

标准数据布局为：

```text
dataset/<DATASET_NAME>/
├── Feature/
│   ├── feature_<subject_id>.npy
│   └── ...
└── Label/
    └── label.npy
```

- 普通特征为 `[N, T, C]`：样本数、时间点数、通道数。
- `label.npy` 为 `[S, 2]`，两列依次为类别标签与受试者 ID。
- 以 `_Aug` 结尾的数据目录会触发增强数据布局；两视图特征布局为 `[2, N, T, C]`，须与对应 loader 和预处理结果一致。
- `--data` 可选 `ADFTD`、`CAEEG`、`CAUEEG`、`IUMSMCI`；`--root_path` 指向包含 `Feature/` 和 `Label/` 的目录。
- `--selected_classes` 用于选择原始类别编号，需依据实际标签含义配置。启用该选项的 loader 会将所选类别重映射为连续编号。

当前 loaders 使用受试者划分；具体比例、分层方式及随机种子见 `data_provider/dataset_loader/`。例如 ADFTD 先划分 20% 测试受试者，再从剩余受试者中划分 20% 作为验证集。

## 训练与测试

请显式指定 `--task_name supervised`。`run.py` 的默认任务名为 `pretraining`，但当前入口没有导入该任务对应的实验类，不能直接使用默认任务运行。

以下以普通、非增强的 19 通道 ADFTD 数据为例；请将路径和通道参数改为实际配置：

```bash
python -u run.py \
  --task_name supervised \
  --is_training 1 \
  --model ADSSTNet \
  --data ADFTD \
  --root_path ./dataset/ADFTD/ \
  --project_name ADFTD-ADSSTNet \
  --d_model 19 \
  --embedding_dim 64 \
  --dim_UVA 4 --topk_A0 4 \
  --d_state 16 --dt_rank 2 \
  --projection_dim 64 \
  --downsample_min_len 4 \
  --sample_adapt_A0 --sample_adapt_Fre \
  --group_size 16 --batch_size 256 \
  --learning_rate 0.001 \
  --optimizer AdamW \
  --train_epochs 200 --patience 15 \
  --itrs 1
```

ADSSTNet 的 `d_model` 应与输入 EEG 通道数匹配。构建模型时，程序从训练数据读取序列长度、输入通道数和类别数量，并基于训练数据计算图先验。批大小与 `group_size` 应结合受试者样本数和显存设置。

只测试时，将 `--is_training` 改为 `0`，其余模型配置、项目名称及迭代设置需与训练保持一致，且对应 checkpoint 必须存在。当前构建流程即使在测试时也会读取训练集，因此仍需完整的数据目录。

更多配置见 `scripts/`，例如：

```bash
bash scripts/ADFTD-Augmentation-NCvsFTDvsAD.sh
bash scripts/ADSSTNet-Ablation-Experiment.sh
```

运行前请检查脚本中的数据路径、`--is_training`、迭代次数及增强设置。部分脚本使用 `--is_training 0`，需要已有权重；脚本中的 `--itr` 是 argparse 对 `--itrs` 的缩写，建议新命令使用完整参数名 `--itrs`。

## 输出与分析

- 权重及训练日志：`checkpoints/<project_name>/itr_XXX/`，主要权重文件为 `checkpoint.pth`。
- 汇总结果：`results/<project_name>/results.txt`。
- 评估包括样本级与受试者级指标；受试者级预测涉及置信度筛选和投票，可通过 `--subject_trim_ratio` 配置。
- `Feature and Visualization Analysis/` 提供 ASM、PGL、RGDI、频带与消融分析 notebooks；使用前需配置本地数据、权重和结果路径。

本次仓库整理未执行模型训练或验证实验，示例命令用于说明现有代码的调用方式。

## 对比模型

实验框架还注册了 EEGNet、Conformer、ADformer、Medformer、STCGRU、TSLANet、MedGNN 和 CodimNet。各模型所需参数请参考其实现及 `run.py`。

## 致谢与许可证

本工程包含基于 [Medformer](https://github.com/DL4mHealth/Medformer) 和 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 等项目的代码。感谢相关作者的开源贡献；使用相应方法时请引用其原始工作。

保留上游 MIT 许可证及版权声明，详见 [LICENSE](LICENSE)。ADSST-Net 的论文信息与引用条目待作者补充。
