---
name: train
description: Use when running or debugging a training run for the U-Net pet segmentation model in this repo — covers common flag combos, where artifacts land, how to read TensorBoard/checkpoint output, and current limitations (no resume-from-checkpoint).
---

# Training the U-Net pet segmentation model

## Running

```bash
pip install -r requirements.txt          # first time only
python main.py --epochs 50 --batch-size 32
```

First run downloads `oxford_iiit_pet` via `tensorflow-datasets` (~800MB) and caches it locally.

## Common flag combos

- **Quick smoke test** (verify the pipeline runs end-to-end, not for real training):
  `python main.py --epochs 1 --batch-size 8 --image-size 64`
- **GPU with mixed precision** (faster, lower memory — only worth it on a GPU with tensor cores):
  `python main.py --mixed-precision`
- **Custom output location**: `python main.py --output-dir /path/to/run`

See `references/config.md` for the full list of tunable fields, including the ones only settable
by editing `src/config.py:Config` directly (not exposed as CLI flags).

## Where output lands

Under `--output-dir` (default `artifacts/`):
- `best_model.keras` — checkpoint with the lowest `val_loss` seen so far (`ModelCheckpoint`,
  `save_best_only=True`). Overwritten whenever a new best is found.
- `logs/` — TensorBoard event files.

View training curves with:

```bash
tensorboard --logdir artifacts/logs
```

## Reading the run

- **`accuracy`** — sparse categorical accuracy, per-pixel.
- **`mean_iou`** — mean intersection-over-union across the 3 classes; a better segmentation
  quality signal than accuracy alone (accuracy can look high just from the background class
  dominating).
- Training stops early if `val_loss` hasn't improved for 8 epochs (`EarlyStopping`,
  `restore_best_weights=True` — the final in-memory model is already the best one seen).
- Learning rate is halved if `val_loss` plateaus for 4 epochs (`ReduceLROnPlateau`, floor `1e-6`).

## Known limitations

- **No resume-from-checkpoint.** `src/train.py:run` always builds a fresh model and starts epoch
  0 — killing a run and restarting does not pick up from `best_model.keras`. If you need this,
  it has to be added (load weights via `keras.models.load_model` before `model.fit`).
- No standalone inference/eval script — see `SPEC.md`'s "Out of scope" section.
