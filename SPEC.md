# SPEC: Pet Segmentation U-Net

## Overview

Semantic segmentation model trained from scratch on the Oxford-IIIT Pet dataset. Given an RGB
image of a cat or dog, the model predicts a per-pixel class label. Built with TensorFlow / Keras,
data loaded via `tensorflow-datasets`.

## Task

- **Input**: RGB image, resized to `image_size x image_size` (default 128x128), normalized to `[0, 1]`.
- **Output**: Per-pixel softmax over `num_classes` (default 3) classes:
  - `0` — pet (foreground)
  - `1` — background
  - `2` — boundary/outline
- **Dataset**: `oxford_iiit_pet` via `tensorflow_datasets`, `train`/`test` splits (~3,680 / ~3,669
  images). First run downloads ~800MB.

## Data pipeline (`src/data.py`)

- Resize images with bilinear interpolation, masks with nearest-neighbor (to avoid interpolating
  label IDs).
- Normalize image pixels to `[0, 1]`; remap mask labels from `1..3` to `0..2`.
- Train split: cache → shuffle (buffer `shuffle_buffer`, seeded) → random horizontal flip
  (image+mask together) → batch → prefetch.
- Val split: resize/normalize → batch → cache → prefetch (no shuffle, no augmentation).

## Model (`src/model.py`)

Standard U-Net, built with the Keras functional API:

- 4 encoder blocks (conv block + max-pool), doubling filters each stage from `base_filters` (32):
  32 → 64 → 128 → 256.
- Bottleneck conv block at 512 filters.
- 4 decoder blocks: `Conv2DTranspose` upsample + skip-connection concat + conv block, mirroring
  the encoder back down to 32 filters.
- Each conv block: two `Conv2D(3x3, same, no bias) → BatchNorm → ReLU`.
- Output head: `Conv2D(num_classes, 1x1, softmax, dtype=float32)` — output forced to float32 so
  the loss stays numerically stable under mixed precision.

## Training (`src/train.py`)

- Optimizer: Adam, configurable learning rate (default `1e-3`).
- Loss: `SparseCategoricalCrossentropy` (masks are integer-labeled, not one-hot).
- Metrics: sparse categorical accuracy, mean IoU.
- Callbacks:
  - `ModelCheckpoint` — saves best model (by `val_loss`) to `<output_dir>/best_model.keras`.
  - `EarlyStopping` — patience 8 on `val_loss`, restores best weights.
  - `ReduceLROnPlateau` — halves LR on `val_loss` plateau (patience 4, min LR `1e-6`).
  - `TensorBoard` — logs to `<output_dir>/logs`.
- Optional mixed precision (`mixed_float16`) via `--mixed-precision`.

## Configuration (`src/config.py`)

Frozen dataclass `Config`, all fields overridable via CLI flags in `main.py`:

| Field | Default | CLI flag |
|---|---|---|
| `dataset_name` | `oxford_iiit_pet` | — |
| `image_size` | `128` | `--image-size` |
| `num_classes` | `3` | — |
| `batch_size` | `32` | `--batch-size` |
| `epochs` | `50` | `--epochs` |
| `learning_rate` | `1e-3` | `--learning-rate` |
| `shuffle_buffer` | `1000` | — |
| `seed` | `42` | — |
| `output_dir` | `artifacts` | `--output-dir` |
| `mixed_precision` | `False` | `--mixed-precision` |

## CLI (`main.py`)

```bash
python main.py --epochs 50 --batch-size 32 [--image-size 128] [--learning-rate 1e-3] \
    [--output-dir artifacts] [--mixed-precision]
```

## Out of scope / not yet implemented

- No inference/prediction script (loading `best_model.keras` and running on new images).
- No evaluation script beyond what `model.fit`'s validation loop reports.
- No test suite.
- No data augmentation beyond horizontal flip (no rotation, crop, color jitter).
