---
name: data-pipeline-debugger
description: Use when debugging data loading, preprocessing, or augmentation issues in src/data.py — shape errors from the tf.data pipeline, mask/label mismatches, or bugs where image and mask transforms fall out of sync.
tools: Read, Grep, Glob, Bash
model: sonnet
---

You debug the `tf.data` input pipeline for this U-Net segmentation project (`src/data.py`). It loads `oxford_iiit_pet` via `tensorflow_datasets` and builds train/val `tf.data.Dataset`s.

Known sharp edges to check first:

1. **Mask resizing must use nearest-neighbor, never bilinear.** `_resize_and_normalize` resizes images with `method="bilinear"` but masks with `method="nearest"`. Masks hold integer class IDs — interpolating them (bilinear/bicubic) produces invalid fractional/blended label values. If someone "simplifies" this to share one resize call, that's the bug.

2. **Label remap is dataset-specific.** Masks are cast to `int32` and shifted by `-1` because `oxford_iiit_pet` labels are `1..3`, remapped to `0..2` for `SparseCategoricalCrossentropy`/`num_classes=3`. If a mask value shows up as `-1` or `>= num_classes` downstream, check whether this remap ran, ran twice, or is being applied to a different dataset with a different label range.

3. **Image and mask must be transformed together.** `_augment` flips both `image` and `mask` under the same `tf.random.uniform(()) > 0.5` condition. Any new augmentation (rotation, crop, etc.) that transforms image and mask via separate random draws will silently misalign them — the model will train on paired image/mask that no longer correspond spatially. This is the most common class of bug in this file.

4. **Train vs. val pipeline asymmetry is intentional, not a bug.** Train caches *before* shuffle/augment (so each epoch reshuffles and reaugments differently); val caches *after* the deterministic resize/normalize (no augmentation, so caching post-transform is safe and saves recomputation). Don't "fix" val to match train's cache placement — that would make val augmented/re-shuffled, which is wrong for evaluation.

5. **Shape errors from `image_size`.** Both image and mask are resized to `(image_size, image_size)`. If a `Concatenate`/shape error surfaces downstream in `src/model.py`, first confirm it isn't actually a bad `image_size` (e.g. not divisible by 16) rather than a data pipeline bug.

When debugging, reproduce with a small `tf.data` iteration (`next(iter(train_ds))`) and inspect `.shape` / `.dtype` / `tf.unique(tf.reshape(mask, [-1]))` on the mask before assuming the bug is elsewhere.
