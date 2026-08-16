import os

from tensorflow import keras

from src.config import Config
from src.data import load_datasets
from src.model import build_unet


def run(config: Config) -> keras.callbacks.History:
    if config.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    os.makedirs(config.output_dir, exist_ok=True)

    train_ds, val_ds, _ = load_datasets(config)

    model = build_unet(config.image_size, config.num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.MeanIoU(num_classes=config.num_classes, sparse_y_pred=False, name="mean_iou"),
        ],
    )

    checkpoint_path = os.path.join(config.output_dir, "best_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        keras.callbacks.TensorBoard(log_dir=os.path.join(config.output_dir, "logs")),
    ]

    return model.fit(train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=callbacks)


if __name__ == "__main__":
    run(Config())
