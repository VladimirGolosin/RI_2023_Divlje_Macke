from tensorflow import keras
from tensorflow.keras import layers

class CatClassifier(keras.Model):
    def __init__(self, num_classes, input_shape):
        super(CatClassifier, self).__init__()
        self.num_classes = num_classes
        self.conv_layers = keras.Sequential([
            layers.Conv2D(4, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2),

        ])

        self.fc_layers = keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        x = self.conv_layers(inputs)
        x = self.fc_layers(x)
        return x
