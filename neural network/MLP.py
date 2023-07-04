import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model

class MLP(Model):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        # Define the input layer
        self.flatten = Flatten(input_shape=(input_size,))

        # Define the hidden layers
        self.hidden_layers = []
        for size in hidden_sizes:
            self.hidden_layers.append(Dense(size, activation='relu'))

        # Define the output layer
        self.output_layer = Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output
