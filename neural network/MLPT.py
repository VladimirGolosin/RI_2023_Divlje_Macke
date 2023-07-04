import torch.nn as nn
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.fc_layers = nn.ModuleList()
        # self.batch_norm_layers = nn.ModuleList()
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout_rate)
        #
        # # Add input layer
        # self.fc_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # self.batch_norm_layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        #
        # # Add hidden layers
        # for i in range(len(hidden_sizes) - 1):
        #     self.batch_norm_layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
        #     self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        #
        # # Add output layer
        # self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.dropout(x)
        y = rearrange(x, 'b c h w -> b (c h w)')
        y = nn.GELU()(self.fc1(y))
        y = nn.GELU()(self.fc2(y))
        y = self.fc3(y)
        return y
        # # Flatten the input tensor
        # x = x.view(x.size(0), -1)
        #
        # # Pass the input through all hidden layers
        # for i, layer in enumerate(self.fc_layers):
        #     x = layer(x)
        #     x = self.batch_norm_layers[i](x)
        #     x = self.relu(x)
        #     x = self.dropout(x)
        #
        # # Pass the output through the output layer
        # x = self.output_layer(x)
        #
        # return x

