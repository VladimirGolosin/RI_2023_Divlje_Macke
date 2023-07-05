import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model

class MLP(Model):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.flatten = Flatten(input_shape=(input_size,))

        self.hidden_layers = []
        for size in hidden_sizes:
            self.hidden_layers.append(Dense(size, activation='relu'))

        self.output_layer = Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output
