# CV Models Fine-Tuning For Binary Classification

Fine-tune computer vision backbones (ResNet-152, Swin Transformer-S, ViT-B/16) on a small two-class food subset (Baked Potato vs Taco). The project uses PyTorch with mixed precision, early stopping, per‑model logging, and simple result plotting.

## Contents
- What this project does
- Project layout
- Prerequisites
- Installation
- Dataset download (Kaggle)
- Configuration (config.yaml)
- Training & evaluation
- Outputs (logs, checkpoints, plots)
- Troubleshooting & notes

## What this project does
- Downloads a food classification dataset from Kaggle automatically on first run (via `kagglehub`).
- Keeps only two classes: "Baked Potato" and "Taco" to make experimentation fast.
- Fine‑tunes three pretrained torchvision models (ResNet-152, Swin-S, ViT-B/16) by replacing the classification head with a 2‑class linear layer.
- Trains with: AdamW, cross‑entropy, optional CUDA mixed precision, early stopping on validation loss.
- Logs training/validation metrics per epoch to `training_logs/<ModelName>.log` and shows loss/accuracy plots at the end.


## Prerequisites
- Python 3.10+ recommended
- A Kaggle account and API credentials (for dataset download)!

## Installation
Windows PowerShell examples below. Adapt paths/commands for your OS and shell.

1) Clone the repo and enter it
```
cd C:\Users\nmago\PycharmProjects\PythonProject\cv-fine-tuning
```

2) Create and activate a virtual environment
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies
```
pip install -r requirements.txt
```

## Dataset download (Kaggle)
The first run will download the dataset automatically via `kagglehub`. Provide your Kaggle API credentials through environment variables:

- KAGGLE_USERNAME
- KAGGLE_KEY

Example (PowerShell):
```bash
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY = "your_api_key"
```

`prepare_dataset.py` will:
- Download the dataset `harishkumardatalab/food-image-classification-dataset`.
- Move it under `./data`.
- Keep only the two classes: `Baked Potato` and `Taco`.
- Apply standard ImageNet normalization and simple flips in the transform pipeline.

Resulting images will reside under:
```
./data/1/Food Classification dataset/{Baked Potato,Taco}/
```

## Configuration (config.yaml)
Key fields you can tune:
- image_size: input resize (default 224)
- learning_rate: AdamW learning rate
- weight_decay: (declared; optimizer currently uses 0.001 internally)
- batch_size: batch size per step
- patience: early stopping patience on validation loss
- seed: random seed (declared; not currently enforced in code)
- num_epochs: max training epochs
- mixed_precision: True/False, enable CUDA autocast + GradScaler
- save_checkpoint: whether to save model weights at the end
- checkpoint_path: path to write checkpoints when save is True

Note: `weight_decay` and `seed` are present in config, but the current `main.py` uses a hardcoded weight decay (0.001) and does not set the seed explicitly.

## Training & evaluation
Run the main script to train all three models sequentially and plot curves:
```
python main.py
```
What happens:
- Dataset is prepared/downloaded on first run.
- Data is split into Train/Val/Test with ratios 0.70/0.15/0.15.
- For each model (ResNet-152, Swin-S, ViT-B/16):
  - Train until early stopping or `num_epochs`.
  - Evaluate on the held‑out test split.
  - Optionally save a checkpoint if `save_checkpoint: True`.
  - Append logs to `training_logs/<ModelName>.log`.
- Two plots are shown per model (loss and accuracy for train/val).

GPU and mixed precision:
- Device is selected automatically: `cuda` if available, otherwise `cpu`.
- If `mixed_precision` is True and a CUDA device is available, autocast/GradScaler are used.

## Outputs
- training_logs/
  - ResNet.log
  - SwinTransformer.log
  - VisionTransformer.log
  (exact filenames match the class names used in `models.py`/`torchvision`)
- Optional model checkpoints under the folder defined by `checkpoint_path` in `config.yaml` if `save_checkpoint: True`.
- Matplotlib windows with train/val loss and accuracy curves for each model.
