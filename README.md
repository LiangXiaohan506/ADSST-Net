# ADSST-Net

ADSST-Net is a research framework for electroencephalogram (EEG) classification. This repository includes model implementations, preprocessing notebooks, supervised training and evaluation, recorded experimental results, and feature interpretation tools. Data loaders are provided for ADFTD, CAEEG, CAUEEG, and IUMSMCI.

## Framework Overview

![Overview of the Proposed Framework](figs/Overview%20of%20the%20Proposed%20Framework.png)

The implementation combines:

- **Adaptive Spectral Modulation (ASM):** adaptive modulation of spectral and channel information.
- **Bidirectional state-space modeling (BiSTS):** Mamba-based sequence modeling.
- **Prior-Guided Graph Learning (PGL):** channel relationship learning initialized with correlation priors computed from training subjects.
- **Gated differential interaction and fusion (RGDI):** interaction, differential suppression, and fusion of sequence and graph representations.
- **Multi-level projection and classification:** classification with configurable instance-level and subject-level contrastive objectives and regularization.

See `models/ADSSTNet.py` and `layers/ADSSTNet_EncDec.py` for implementation details.

## Repository Structure

```text
ADSST-Net/
├── run.py                              # Experiment entry point
├── models/                             # ADSSTNet and baseline models
├── layers/                             # Model components
├── exp/                                # Training, validation, and testing
├── data_provider/                      # Data loaders and sampling
├── data_preprocessing/                 # Preprocessing and partitioning notebooks
├── scripts/                            # Experiment and ablation configurations
├── Feature and Visualization Analysis/ # Interpretation and visualization notebooks
├── utils/                              # Losses, metrics, and utilities
├── figs/                               # Framework figure and plotting assets
├── dataset/
│   └── .gitkeep                        # Placeholder; download data separately
├── checkpoints/
│   └── README.md                       # Checkpoint download information
├── results/
│   ├── results.xlsx                    # Existing results workbook
│   └── <project_name>/results.txt       # Recorded experiment outputs
├── requirements.txt
└── LICENSE
```

Raw/processed datasets and model weights are not included. The `dataset/` directory contains only an empty `.gitkeep` file because Git does not track empty directories.

## Public Datasets

- **CAUEEG:** The CAUEEG dataset is publicly available at the [official CAUEEG repository](https://github.com/ipis-mjkim/caueeg-dataset).
- **ADFTD:** The ADFTD datasets are available from OpenNeuro: [ds004504, version 1.0.8](https://openneuro.org/datasets/ds004504/versions/1.0.8) and [ds006036, version 1.0.5](https://openneuro.org/datasets/ds006036/versions/1.0.5).
- **IUMSMCI:** The IUMSMCI dataset is publicly available from the Medical Image and Signal Processing Research Center: [English page](https://misp.mui.ac.ir/en/eeg-data-0) and [Persian page](https://misp.mui.ac.ir/fa/eeg-data).

Follow each provider's access instructions, terms of use, and citation requirements. Use the corresponding preprocessing notebook to convert downloaded recordings to the layout expected by this project. A CAEEG loader and preprocessing notebooks are also included; no public CAEEG download link is provided here.

## Environment and Installation

The original environment records **Python 3.10.14** and **PyTorch 2.5.1 with CUDA 11.8**. A Linux environment with an NVIDIA GPU is recommended for the Mamba/Triton dependencies; macOS and CPU-only execution have not been verified.

```bash
git clone https://github.com/LiangXiaohan506/ADSST-Net.git
cd ADSST-Net
conda create -n adsstnet python=3.10.14 -y
conda activate adsstnet

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Exclude the Python version record and the separately installed PyTorch entry.
python -c "from pathlib import Path; p=Path('requirements.txt'); Path('requirements-runtime.txt').write_text('\n'.join(s for s in p.read_text().splitlines() if not s.strip().startswith(('python==', 'torch')))+'\n')"
pip install -r requirements-runtime.txt

# Additional packages imported by the source code.
pip install timm wandb visdom
pip install causal-conv1d mamba-ssm --no-build-isolation
```

Mamba and its CUDA extensions must match your PyTorch, CUDA, and compiler environment. Additional dependencies above are not version-pinned, and a clean installation has not been verified. Jupyter is needed for notebooks; `thop` is optional for FLOPs/MACs reporting.

## Data Preparation

Configure local input paths in the notebooks under `data_preprocessing/`, then place processed data under:

```text
dataset/<DATASET_NAME>/
├── Feature/
│   ├── feature_<subject_id>.npy
│   └── ...
└── Label/
    └── label.npy
```

- Standard features have shape `[N, T, C]`: samples, time points, and channels.
- `label.npy` has shape `[S, 2]`: class label and subject ID for each subject.
- Directory names ending in `_Aug` activate the augmented-data layout. Two-view features use `[2, N, T, C]`; ensure preprocessing and loader settings agree.
- Set `--data` to `ADFTD`, `CAEEG`, `CAUEEG`, or `IUMSMCI`, and set `--root_path` to the directory containing `Feature/` and `Label/`.
- Configure `--selected_classes` according to the labels in your data. Loaders supporting this option remap selected labels to consecutive indices.

The loaders use subject-level partitions. Refer to `data_provider/dataset_loader/` for the exact split and stratification settings. For example, the ADFTD loader reserves 20% of subjects for testing, then allocates 20% of the remaining subjects to validation.

## Training

Always specify `--task_name supervised`. The default task in `run.py` is `pretraining`, but its experiment class is not imported by the current entry point.

Example for ordinary, non-augmented, 19-channel ADFTD data; adjust the path and channel configuration to match your data:

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

For ADSSTNet, `d_model` should match the EEG channel count. Model construction reads sequence length, channel count, and class count from the training data, and computes graph priors from that data. Choose `batch_size` and `group_size` according to available samples per subject and GPU memory.

Additional experiment configurations are provided in `scripts/`, including:

```bash
bash scripts/ADFTD-Augmentation-NCvsFTDvsAD.sh
bash scripts/ADSSTNet-Ablation-Experiment.sh
```

Review paths, augmentation settings, iteration count, and `--is_training` before running. Some scripts use `--is_training 0` and require existing weights. Their `--itr` argument is an argparse abbreviation of `--itrs`; use the full spelling in new commands.

## Checkpoints and Evaluation

Checkpoint download information is available in [checkpoints/README.md](checkpoints/README.md). The cloud-storage link currently contains the placeholder `*********` and must be replaced with the actual download URL.

Place downloaded weights in the following layout:

```text
checkpoints/<project_name>/itr_000/checkpoint.pth
checkpoints/<project_name>/itr_001/checkpoint.pth
...
```

To evaluate an existing checkpoint, use the training command with `--is_training 0`. Keep the architecture parameters, `project_name`, and iteration settings consistent with the saved run. The current evaluation initialization also reads training data, so the complete processed dataset is still required.

## Experimental Results

Existing local experiment outputs are included in [results/](results/), preserving their original filenames and contents:

- [results/results.xlsx](results/results.xlsx): the existing results workbook.
- `results/<project_name>/results.txt`: per-experiment records, including sample-level and subject-level metrics and confusion matrices where available.

Directory names identify the dataset, model, classification task, and augmentation configuration. Consult each record for its actual iterations and metrics; files can contain multiple recorded runs. These are uploaded experiment artifacts, and no new training or evaluation was performed for this repository update.

Training logs and weights are written to `checkpoints/<project_name>/itr_XXX/`. Subject-level evaluation includes confidence filtering and voting, configurable through `--subject_trim_ratio`.

## Feature Analysis and Baselines

`Feature and Visualization Analysis/` contains notebooks for ASM, PGL, RGDI, frequency-band analysis, and ablation visualization. Configure local data, checkpoint, and result paths before use.

The experiment framework also registers EEGNet, Conformer, ADformer, Medformer, STCGRU, TSLANet, MedGNN, and CodimNet. Refer to `run.py` and the corresponding model implementations for parameters.

## Acknowledgments and License

This project includes code based on [Medformer](https://github.com/DL4mHealth/Medformer), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), and related open-source work. Please cite the original methods and datasets when using them.

The upstream MIT license and copyright notice are retained in [LICENSE](LICENSE). ADSST-Net publication details and a citation entry will be added when available.
