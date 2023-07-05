import torch.nn as nn
from einops import rearrange
import torch
import math

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.w1 = nn.Parameter(torch.empty((hidden_sizes, input_size)))
        self.b1 = nn.Parameter(torch.empty((hidden_sizes,)))
        self.bn1 = nn.BatchNorm1d(hidden_sizes)

        self.w2 = nn.Parameter(torch.empty((hidden_sizes, hidden_sizes)))
        self.b2 = nn.Parameter(torch.empty((hidden_sizes,)))
        self.bn2 = nn.BatchNorm1d(hidden_sizes)

        self.w3 = nn.Parameter(torch.empty((output_size, hidden_sizes)))
        self.b3 = nn.Parameter(torch.empty((output_size,)))

        self.dropout = nn.Dropout(p=dropout_rate)

        self.reset_parameters()


    def forward(self, x):
        x = self.dropout(x)
        y = rearrange(x, 'b c h w -> b (c h w)')
        y = nn.GELU()(self.bn1(y @ self.w1.T + self.b1))
        y = nn.GELU()(y)
        y = nn.GELU()(self.bn2(y @ self.w2.T + self.b2))
        y = nn.GELU()(y)
        y = y @ self.w3.T + self.b3
        return y


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.constant_(self.b1, 0)
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.constant_(self.b2, 0)
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        nn.init.constant_(self.b3, 0)

