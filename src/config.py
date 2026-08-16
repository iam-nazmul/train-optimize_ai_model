from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    dataset_name: str = "oxford_iiit_pet"
    image_size: int = 128
    num_classes: int = 3
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    shuffle_buffer: int = 1000
    seed: int = 42
    output_dir: str = "artifacts"
    mixed_precision: bool = False
