import tensorflow as tf
from tensorflow.keras import layers, models


def BaseBlockFunctional(x, d_model):
    dim = d_model // 4

    # Bottleneck
    bottleneck = layers.Conv1D(dim, kernel_size=1, strides=1, use_bias=True)(x)

    # Max pooling path
    x1 = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = layers.Conv1D(dim, kernel_size=1, strides=1, use_bias=True)(x1)

    # Convolutional paths
    x2 = layers.Conv1D(dim, kernel_size=3, strides=1, padding='same', use_bias=True)(bottleneck)
    x3 = layers.Conv1D(dim, kernel_size=5, strides=1, padding='same', use_bias=True)(bottleneck)
    x4 = layers.Conv1D(dim, kernel_size=7, strides=1, padding='same', use_bias=True)(bottleneck)

    # Concatenate
    x_out = layers.Concatenate()([x1, x2, x3, x4])

    # Batch Normalization and ReLU
    # x_out = layers.BatchNormalization()(x_out)
    x_out = layers.ReLU()(x_out)

    # # Pooling
    # x_out = layers.MaxPooling1D(pool_size=2)(x_out)

    return x_out


def MiniInceptionTimeFunctional(input_shape, in_channel=1, d_model=64, num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)

    # Projection layer
    x = layers.Conv1D(d_model, kernel_size=7, strides=1, padding='same', use_bias=True)(inputs)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2, 2)(x)
    # x = layers.BatchNormalization()(x)

    # Shortcuts
    shortcut1 = x

    # BaseBlock 1 and 2
    x = BaseBlockFunctional(x, d_model)
    x = BaseBlockFunctional(x, d_model)

    # First addition with shortcut1
    x = layers.Add()([x, shortcut1])

    # Projection layer
    x = layers.Conv1D(2 * d_model, kernel_size=5, strides=2, padding='same', use_bias=True)(x)
    x = layers.ReLU()(x)
    # x = layers.MaxPooling1D(2, 2)(x)

    # Shortcuts
    shortcut2 = x

    # BaseBlock 3 and 4
    x = BaseBlockFunctional(x, 2 * d_model)
    x = BaseBlockFunctional(x, 2 * d_model)

    # Second addition with shortcut2
    x = layers.Add()([x, shortcut2])

    # Projection layer
    x = layers.Conv1D(4 * d_model, kernel_size=3, strides=2, padding='same', use_bias=True)(x)
    x = layers.ReLU()(x)
    # x = layers.MaxPooling1D(2, 2)(x)

    # Shortcuts
    shortcut3 = x

    # BaseBlock 3 and 4
    x = BaseBlockFunctional(x, 4 * d_model)
    x = BaseBlockFunctional(x, 4 * d_model)

    # Second addition with shortcut2
    x = layers.Add()([x, shortcut3])

    # Global Average Pooling and Fully Connected Layer
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = MiniInceptionTimeFunctional(input_shape=(320, 1), d_model=8, num_classes=5)
    model.build((None, 320, 1))
    model.summary()
