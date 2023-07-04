import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.relu = nn.ReLU()

        # Add input layer
        self.fc_layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Add output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)

        # Pass the input through all hidden layers
        for layer in self.fc_layers:
            x = self.relu(layer(x))

        # Pass the output through the output layer
        x = self.output_layer(x)

        return x
