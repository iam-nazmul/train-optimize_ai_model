import tensorflow as tf
import tensorflow_datasets as tfds

from src.config import Config

AUTOTUNE = tf.data.AUTOTUNE


def _resize_and_normalize(example: dict, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(example["image"], (image_size, image_size), method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.image.resize(example["segmentation_mask"], (image_size, image_size), method="nearest")
    mask = tf.cast(mask, tf.int32) - 1  # dataset labels are 1..3 -> map to 0..2

    return image, mask


def _augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask


def load_datasets(config: Config) -> tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    (train_ds, val_ds), info = tfds.load(
        config.dataset_name,
        split=["train", "test"],
        with_info=True,
    )

    train_ds = (
        train_ds.map(lambda ex: _resize_and_normalize(ex, config.image_size), num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(config.shuffle_buffer, seed=config.seed)
        .map(_augment, num_parallel_calls=AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds.map(lambda ex: _resize_and_normalize(ex, config.image_size), num_parallel_calls=AUTOTUNE)
        .batch(config.batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, info
