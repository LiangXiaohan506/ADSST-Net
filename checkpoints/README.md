# Pretrained Checkpoints

Cloud-storage download link for the checkpoint files:

[checkpoints](https://drive.google.com/file/d/1455jL941Uf9mktoHBdjMZq5J32ZnwmbW/view?usp=drive_link)

The link above is a placeholder and will be replaced with the actual download URL.

Download and extract the checkpoint files into this directory, preserving the experiment and iteration structure:

```text
checkpoints/
├── README.md
└── <project_name>/
    ├── itr_000/
    │   └── checkpoint.pth
    ├── itr_001/
    │   └── checkpoint.pth
    └── ...
```

Use the same model configuration, project name, and iteration count as the saved experiment. Set `--task_name supervised --is_training 0` for evaluation. The current model initialization still requires the processed training data.

Model weight files are downloaded separately and are not stored in this Git repository.
