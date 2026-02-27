# SegMate

An efficient 2.5D framework for multi-organ medical image segmentation.

## Abstract

State-of-the-art models for medical image segmentation achieve excellent accuracy but require substantial computational resources, limiting deployment in resource-constrained clinical settings. We present SegMate, an efficient 2.5D framework that achieves state-of-the-art accuracy, while considerably reducing computational requirements. Our efficient design is the result of meticulously integrating asymmetric architectures, attention mechanisms, multi-scale feature fusion, slice-based positional conditioning, and multi-task optimization. We demonstrate the efficiency-accuracy trade-off of our framework across three modern backbones (EfficientNetV2-M, MambaOut-Tiny, FastViT-T12). We perform experiments on three datasets: TotalSegmentator, SegTHOR and AMOS22. Compared with the vanilla models, SegMate reduces computation (GFLOPs) by up to 2.5× and memory footprint (VRAM) by up to 2.1×, while generally registering performance gains of around 1%. On TotalSegmentator, we achieve a Dice score of 93.51% with only 295MB peak GPU memory. Zero-shot cross-dataset evaluations on SegTHOR and AMOS22 demonstrate strong generalization, with Dice scores of up to 86.85% and 89.35%, respectively. 

## Repository Structure

```
SegMate/
├── models/                # Model architectures
│   ├── segmate_v2.py              # SegMate with EfficientNetV2 encoder (primary)
│   ├── segmate_v2_film.py         # SegMate with FiLM slice-position conditioning
│   ├── segmate_mambaout.py        # SegMate with MambaOut encoder
│   ├── segmate_mambaout_film.py   # MambaOut + FiLM conditioning
│   ├── segmate_fastvit.py         # SegMate with FastViT encoder
│   ├── segmate_fastvit_film.py    # FastViT + FiLM conditioning
│   ├── vanilla_segmate_v2.py      # Ablation: no SE/CBAM attention
│   ├── segmate_v2_nose.py         # Ablation: no SE blocks
│   └── ...
├── train/                 # Training scripts
│   ├── train_segmatev2_25D.py     # Train SegMate (EfficientNetV2, configurable encoder)
│   ├── train_segmate_25D.py       # Train SegMate (original architecture)
│   ├── train_segmate_mambaout_25D.py
│   ├── train_segmate_fastvit_25D.py
│   ├── finetune_25D.py            # Fine-tune on SegTHOR / AMOS22
│   └── ...
├── evaluation/            # Evaluation and metrics
│   ├── metrics_evaluation_25D.py         # Evaluate on TotalSegmentator
│   ├── metrics_evaluation_segthor_25D.py # Evaluate on SegTHOR
│   ├── metrics_evaluation_amos22_25D.py  # Evaluate on AMOS22
│   ├── benchmark_models.py               # FLOPs / memory benchmarking
│   └── ...
├── scripts/               # Dataset preparation
│   ├── improved_totalsegmentator_dataset.py  # Prepare TotalSegmentator
│   ├── prepare_segthor_dataset.py            # Prepare SegTHOR
│   ├── prepare_amos22_dataset.py             # Prepare AMOS22
│   └── inference.py                          # Run inference on new data
├── losses/                # Loss functions (Dice, Focal, Boundary, Presence)
│   ├── combined_loss.py
│   └── custom_unetpp_loss.py
├── dataloaders/           # 2.5D balanced dataloaders
│   ├── balanced_dataloader_25D.py
│   ├── balanced_dataloader_25D_film.py
│   └── ...
└── utils/                 # Augmentations and data loading utilities
    ├── augmentations.py
    ├── segthor_augmentations.py
    ├── amos22_augmentations.py
    └── data_loader.py
```

## Prerequisites

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- Key dependencies:

```
torch
torchvision
timm
segmentation-models-pytorch
nibabel
medpy
scipy
numpy
pandas
scikit-learn
scikit-image
albumentations
kornia
tqdm
tensorboard
fvcore
```

Install all dependencies:

```bash
pip install torch torchvision timm segmentation-models-pytorch \
    nibabel medpy scipy numpy pandas scikit-learn scikit-image \
    albumentations kornia tqdm tensorboard fvcore
```

## Data Preparation

All preparation scripts convert 3D NIfTI volumes into 2.5D chunked slices (default chunk size: 8 adjacent axial slices) and split them into train/val/test sets. The output directories are self-contained and ready for training.

### TotalSegmentator

Download the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) dataset (v1), then run:

```bash
python scripts/improved_totalsegmentator_dataset.py --root /path/to/TotalSegmentator
```

Output: `processed_dataset/` with train/val/test splits (80/10/10).

Optional flags:
- `--chunk_size 8` -- number of adjacent slices per sample (default: 8)
- `--min_foreground 0.001` -- minimum foreground ratio to keep a slice
- `--resize 256 256` -- resize slices to a fixed spatial resolution
- `--limit 10` -- process only N patients (for debugging)

### SegTHOR

Download the [SegTHOR](https://competitions.codalab.org/competitions/21145) dataset, then run:

```bash
python scripts/prepare_segthor_dataset.py --root /path/to/SegTHOR
```

Output: `processed_dataset_segthor/` (30 train / 5 val / 5 test patients). The script maps SegTHOR labels (esophagus, aorta, trachea) into the shared 9-channel format used by TotalSegmentator, so all models are directly compatible across datasets.

### AMOS22

Download the [AMOS22](https://amos22.grand-challenge.org/) dataset, then run:

```bash
python scripts/prepare_amos22_dataset.py --root /path/to/amos22
```

Output: `processed_dataset_amos22/`. Only CT scans (patient IDs < 500) are used. The script maps AMOS22 labels (esophagus, liver, kidneys, aorta) into the shared 9-channel format.

## Training

### Single-GPU training (EfficientNetV2 encoder)

```bash
python train/train_segmatev2_25D.py \
    --root processed_dataset \
    --encoder tf_efficientnetv2_m \
    --epochs 100 \
    --batch_size 32 \
    --deep_supervision \
    --logdir runs/segmate_effv2m
```

Available `--encoder` options: `tf_efficientnetv2_s`, `tf_efficientnetv2_m`, `tf_efficientnetv2_l` (or any timm-compatible encoder name). Pass `--no_pretrained` to train from scratch without ImageNet weights.

### Backbone-specific training scripts

For MambaOut and FastViT backbones, use the dedicated training scripts:

```bash
# MambaOut encoder
python train/train_segmate_mambaout_25D.py \
    --root processed_dataset \
    --epochs 100 \
    --batch_size 32 \
    --deep_supervision \
    --logdir runs/segmate_mambaout

# FastViT encoder
python train/train_segmate_fastvit_25D.py \
    --root processed_dataset \
    --epochs 100 \
    --batch_size 32 \
    --deep_supervision \
    --logdir runs/segmate_fastvit
```

### Multi-GPU training (DDP)

Wrap any training script with `torchrun`:

```bash
torchrun --nproc_per_node=4 train/train_segmatev2_25D.py \
    --root processed_dataset \
    --encoder tf_efficientnetv2_m \
    --epochs 100 \
    --batch_size 32 \
    --deep_supervision \
    --logdir runs/segmate_effv2m_ddp
```

### Resuming from a checkpoint

All training scripts accept `--resume`:

```bash
python train/train_segmatev2_25D.py \
    --root processed_dataset \
    --encoder tf_efficientnetv2_m \
    --epochs 100 \
    --batch_size 32 \
    --deep_supervision \
    --resume checkpoints/segmate_v2_epoch50.pth \
    --logdir runs/segmate_effv2m
```

### Monitoring

Training logs are written to TensorBoard. Launch the dashboard with:

```bash
tensorboard --logdir runs/
```

## Fine-tuning

Fine-tune a TotalSegmentator-pretrained model on SegTHOR or AMOS22 using the unified fine-tuning script. The script freezes the encoder for the first few epochs to prevent catastrophic forgetting, then uses a cosine-annealing schedule with a low learning rate.

### Fine-tune on SegTHOR

```bash
python train/finetune_25D.py \
    --model_type SegMateV2 \
    --encoder_name tf_efficientnetv2_m \
    --pretrained_checkpoint checkpoints/segmate_v2_best.pth \
    --root processed_dataset_segthor \
    --epochs 25 \
    --batch_size 32 \
    --deep_supervision \
    --freeze_encoder_epochs 5 \
    --checkpoint_dir segthor_checkpoints
```

### Fine-tune on AMOS22

```bash
python train/finetune_25D.py \
    --model_type SegMateV2 \
    --encoder_name tf_efficientnetv2_m \
    --pretrained_checkpoint checkpoints/segmate_v2_best.pth \
    --root processed_dataset_amos22 \
    --epochs 25 \
    --batch_size 32 \
    --deep_supervision \
    --freeze_encoder_epochs 5 \
    --checkpoint_dir amos22_checkpoints
```

Supported `--model_type` values: `SegMate`, `SegMateV2`, `SegMateFiLM`, `SegMateMambaOut`, `SegMateMambaOutFiLM`, `SegMateFastViT`, `SegMateFastViTFiLM`, `VanillaSegMateMambaOut`, `VanillaSegMateFastViT`.

Additional fine-tuning options:
- `--lr 5e-6` -- learning rate (default: 5e-6, intentionally low)
- `--freeze_encoder_epochs 5` -- freeze encoder for first N epochs (default: 5)
- `--early_stop_patience 10` -- stop if no improvement for N epochs (0 to disable)

## Evaluation

Each dataset has a dedicated evaluation script that computes per-organ Dice, HD95, IoU, Precision, and Recall, and writes a summary CSV.

### TotalSegmentator

```bash
python evaluation/metrics_evaluation_25D.py \
    --model_path checkpoints/segmate_v2_best.pth \
    --root_dir processed_dataset \
    --split test \
    --deep_supervision
```

### SegTHOR

```bash
python evaluation/metrics_evaluation_segthor_25D.py \
    --model_path segthor_checkpoints/best.pth \
    --root_dir processed_dataset_segthor \
    --split test \
    --deep_supervision
```

### AMOS22

```bash
python evaluation/metrics_evaluation_amos22_25D.py \
    --model_path amos22_checkpoints/best.pth \
    --root_dir processed_dataset_amos22 \
    --split test \
    --deep_supervision
```

### Model benchmarking

Measure FLOPs and peak GPU memory for any model:

```bash
python evaluation/benchmark_models.py
```

## Supported Models

| Architecture | Encoder | Key feature |
|---|---|---|
| SegMate | Custom (original) | SE + CBAM dual attention |
| SegMateV2 | EfficientNetV2 (S/M/L) | Configurable timm encoder |
| SegMateFiLM | EfficientNetV2 | FiLM slice-position conditioning |
| SegMateMambaOut | MambaOut-Tiny | State-space model backbone |
| SegMateMambaOutFiLM | MambaOut-Tiny | MambaOut + FiLM conditioning |
| SegMateFastViT | FastViT-T12 | Hybrid CNN-Transformer backbone |
| SegMateFastViTFiLM | FastViT-T12 | FastViT + FiLM conditioning |
| Vanilla variants | (various) | Ablation models without SE/CBAM |
| NoSE variants | (various) | Ablation models without SE blocks |

## Target Organs

All models segment the same 9 anatomical structures (in channel order):

1. Left Lung
2. Right Lung
3. Spinal Cord
4. Esophagus
5. Liver
6. Left Kidney
7. Right Kidney
8. Aorta
9. Trachea

## License

This repository is provided for research purposes.
