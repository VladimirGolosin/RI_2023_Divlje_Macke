import torch.nn as nn
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn3 = nn.BatchNorm1d(hidden_sizes)
        self.fc4 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn4 = nn.BatchNorm1d(hidden_sizes)
        self.fc5 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn5 = nn.BatchNorm1d(hidden_sizes)
        self.fc6 = nn.Linear(hidden_sizes, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.fc_layers = nn.ModuleList()
        # self.batch_norm_layer = nn.BatchNorm1d(hidden_sizes)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout_rate)
        #
        # # Add input layer
        # self.fc_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # self.batch_norm_layers.append(nn.BatchNorm1d(hidden_sizes))
        #
        # # Add hidden layers
        # for i in range(6):
        #     self.batch_norm_layers.append(nn.BatchNorm1d(hidden_sizes))
        #     self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        #
        # # Add output layer
        # self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.dropout(x)
        y = rearrange(x, 'b c h w -> b (c h w)')
        y = nn.GELU()(self.bn1(self.fc1(y)))
        y = nn.GELU()(self.bn2(self.fc2(y)))
        y = nn.GELU()(self.bn3(self.fc3(y)))
        y = nn.GELU()(self.bn4(self.fc4(y)))
        y = nn.GELU()(self.bn5(self.fc5(y)))
        y = self.fc6(y)
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

