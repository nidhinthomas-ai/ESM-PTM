import torch
import torch.nn as nn
from torch.nn.functional import relu

class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        filters = 128
        kernels = 5
        dense_layers1 = 64
        dense_layers2 = 32
        dropout = 0.2

        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=filters, kernel_size=kernels, padding='same')
        self.conv1_bn = nn.BatchNorm1d(filters)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters * input_shape[0], dense_layers1)
        self.fc2 = nn.Linear(dense_layers1, dense_layers2)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(dense_layers2, 1)

    def forward(self, x):
        x = relu(self.conv1_bn(self.conv1(x)))
        x = self.flatten(x)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
