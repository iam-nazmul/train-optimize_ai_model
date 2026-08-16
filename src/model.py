from tensorflow import keras
from keras import layers


def _conv_block(x: keras.KerasTensor, filters: int) -> keras.KerasTensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def _encoder_block(x: keras.KerasTensor, filters: int) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    skip = _conv_block(x, filters)
    pooled = layers.MaxPooling2D(2)(skip)
    return skip, pooled


def _decoder_block(x: keras.KerasTensor, skip: keras.KerasTensor, filters: int) -> keras.KerasTensor:
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = _conv_block(x, filters)
    return x


def build_unet(image_size: int, num_classes: int, base_filters: int = 32) -> keras.Model:
    inputs = keras.Input(shape=(image_size, image_size, 3))

    skip1, x = _encoder_block(inputs, base_filters)
    skip2, x = _encoder_block(x, base_filters * 2)
    skip3, x = _encoder_block(x, base_filters * 4)
    skip4, x = _encoder_block(x, base_filters * 8)

    x = _conv_block(x, base_filters * 16)

    x = _decoder_block(x, skip4, base_filters * 8)
    x = _decoder_block(x, skip3, base_filters * 4)
    x = _decoder_block(x, skip2, base_filters * 2)
    x = _decoder_block(x, skip1, base_filters)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs, name="unet")
