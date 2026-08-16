# Config field reference

Source of truth: `src/config.py:Config` (frozen dataclass). Fields marked "CLI" are wired into
`main.py`'s argparse; the rest are code-only — change them by editing `Config`'s defaults or by
constructing a `Config(...)` yourself instead of going through `main.py`.

| Field | Default | CLI flag |
|---|---|---|
| `dataset_name` | `oxford_iiit_pet` | — (code-only) |
| `image_size` | `128` | `--image-size` |
| `num_classes` | `3` | — (code-only; changing this also requires touching the label remap in `src/data.py` and `mean_iou`'s `num_classes` in `src/train.py`) |
| `batch_size` | `32` | `--batch-size` |
| `epochs` | `50` | `--epochs` |
| `learning_rate` | `1e-3` | `--learning-rate` |
| `shuffle_buffer` | `1000` | — (code-only) |
| `seed` | `42` | — (code-only) |
| `output_dir` | `artifacts` | `--output-dir` |
| `mixed_precision` | `False` | `--mixed-precision` (flag, no value) |

## Notes

- `image_size` should stay divisible by 16: the U-Net has 4 encoder max-pool stages
  (`src/model.py`), each halving spatial resolution, and the decoder upsamples back by the same
  factor via `Conv2DTranspose`. A size not divisible by `2^4` can produce an off-by-one mismatch
  between an encoder skip tensor and its corresponding decoder upsample output at the
  `Concatenate` step.
- `num_classes` is threaded through three places that must stay in sync: `src/model.py`'s output
  `Conv2D` filter count, `src/data.py`'s mask label remap (currently hardcoded to the dataset's
  `1..3 → 0..2`, which is specific to `num_classes=3`), and `src/train.py`'s
  `MeanIoU(num_classes=...)` metric.
