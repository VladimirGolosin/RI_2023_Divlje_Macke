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
