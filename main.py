import argparse
import dataclasses

from src.config import Config
from src.train import run


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train a U-Net segmentation model from scratch.")
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--mixed-precision", action="store_true", default=defaults.mixed_precision)
    args = parser.parse_args()

    return dataclasses.replace(
        defaults,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
    )


if __name__ == "__main__":
    run(parse_args())
