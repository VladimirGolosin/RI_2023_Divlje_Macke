import tensorflow as tf
from tensorflow.keras import layers

class MLP(tf.keras.Model):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.fc_layers = []
        self.relu = tf.keras.activations.relu

        # Add input layer
        self.fc_layers.append(layers.Dense(hidden_sizes[0], input_shape=(input_size,), activation='relu'))

        # Add hidden layers
        for i in range(1, len(hidden_sizes)):
            self.fc_layers.append(layers.Dense(hidden_sizes[i], activation='relu'))

        # Add output layer
        self.output_layer = layers.Dense(output_size, activation='softmax')

    def call(self, x):
        # Flatten the input tensor
        x = tf.keras.layers.Flatten()(x)

        # Pass the input through all hidden layers
        for layer in self.fc_layers:
            x = layer(x)

        # Pass the output through the output layer
        x = self.output_layer(x)

        return x
