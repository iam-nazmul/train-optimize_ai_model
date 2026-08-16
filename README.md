# ai/ml train-optimize_ai_model

U-Net image segmentation trained from scratch on the Oxford-IIIT Pet dataset (TensorFlow / Keras).

## Setup

Requires Python 3.9–3.12 — TensorFlow doesn't yet publish wheels for newer versions. If your
default `python3` is newer than that (check with `python3 --version`), install 3.12 first
(e.g. `brew install python@3.12` on macOS) and use it to create the virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python main.py --epochs 50 --batch-size 32
```

First run downloads the dataset via `tensorflow-datasets` (~800MB). Checkpoints, TensorBoard
logs, and the best model land in `artifacts/` (override with `--output-dir`).

## Contributing

See `src/README.md` for module-level developer docs and `CLAUDE.md` for the architecture overview.
