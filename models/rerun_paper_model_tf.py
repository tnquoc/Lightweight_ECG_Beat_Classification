import tensorflow as tf
from tensorflow.keras import layers, models, initializers


def CustomModel(input_shape, in_channels, first_width, num_classes):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Block 0
    x = layers.Conv1D(first_width, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv1D(first_width, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Block 1
    x = layers.Conv1D(first_width * 2, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(first_width * 2, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Block 2
    x = layers.Conv1D(first_width * 4, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(first_width * 4, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Block 3
    x = layers.Conv1D(first_width * 8, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(first_width * 8, kernel_size=5, strides=1, padding='same',
                      kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalAveragePooling1D()(x)  # Equivalent to AdaptiveAvgPool1d(1)

    # Classifier
    x = layers.Dense(20, kernel_initializer=initializers.HeNormal())(x)
    x = layers.ReLU()(x)
    # x = layers.Dropout(0.3)(x)  # Uncomment this line if you need dropout
    outputs = layers.Dense(num_classes)(x)

    # Define model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':
    model = CustomModel(input_shape=(320, 1), in_channels=1, first_width=4, num_classes=5)
    model.build((None, 320, 1))
    model.summary()
