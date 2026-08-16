# src/ — developer notes

Audience: developers/agents working on this codebase. For task definition and hyperparameter
defaults, see `../SPEC.md`. For repo-wide rules, see `../CLAUDE.md`.

## `config.py`

`Config` is a frozen dataclass — the single source of truth for every hyperparameter. `main.py`
builds one via `Config()` defaults and `dataclasses.replace(...)` from parsed CLI args. Not every
field is CLI-exposed (`dataset_name`, `num_classes`, `shuffle_buffer`, `seed` are code-only) — see
`.claude/skills/train/references/config.md` for the full field table and which fields are CLI vs
code-only.

## `data.py`

`load_datasets(config)` builds the train/val `tf.data.Dataset`s from `oxford_iiit_pet`
(`tensorflow_datasets`).

- `_resize_and_normalize`: images resized bilinear + normalized to `[0, 1]`; masks resized
  **nearest-neighbor** (they hold integer class IDs — bilinear would corrupt them) and remapped
  from the dataset's `1..3` labels to `0..2`.
- `_augment`: random horizontal flip, applied to image and mask under the *same* draw so they stay
  spatially aligned. Any new augmentation must follow this pattern.
- Train pipeline caches **before** shuffle/augment (so augmentation varies per epoch). Val caches
  **after** the deterministic resize/normalize, with no shuffle/augmentation — this asymmetry is
  intentional, not a bug.

## `model.py`

`build_unet(image_size, num_classes, base_filters=32)` — functional-API U-Net.

- 4 encoder blocks (conv block + max-pool), filters doubling each stage: `base_filters` →
  `base_filters*2` → `*4` → `*8`.
- Bottleneck conv block at `base_filters*16`.
- 4 decoder blocks mirror the encoder: `Conv2DTranspose` upsample → concat with the matching
  encoder skip tensor → conv block.
- Output `Conv2D(num_classes, 1, activation="softmax", dtype="float32")` — the `dtype="float32"`
  pin overrides the global mixed-precision policy so softmax/loss stay numerically stable when
  `--mixed-precision` is set.

## `train.py`

`run(config)` compiles and fits the model.

- Loss is `SparseCategoricalCrossentropy` because masks from `data.py` are integer-labeled, not
  one-hot.
- Metrics: sparse categorical accuracy, `MeanIoU` (a better segmentation-quality signal than
  accuracy, which can look high purely from background-class dominance).
- Callbacks: `ModelCheckpoint` (best `val_loss` → `<output_dir>/best_model.keras`),
  `EarlyStopping` (patience 8, restores best weights), `ReduceLROnPlateau` (patience 4, floor
  `1e-6`), `TensorBoard` (→ `<output_dir>/logs`).
- No checkpoint-resume: `run()` always starts a fresh model at epoch 0.
