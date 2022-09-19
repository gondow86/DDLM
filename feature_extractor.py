import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

n_features = 128
n_epochs = 800
batch_size = 96
beta = 1
gamma = 1
lambda_0 = 0.1
lambda_1 = 0.1
learning_rate = 0.0001


def min_max_normalize(x):
    x = (x - x.min()) / (x.max() - x.min())  # x: ndarray
    return x


class ConvNet1D(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_input, n_hidden1, kernel_size=3),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_hidden1, n_hidden2, kernel_size=3),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_hidden2, n_output, kernel_size=24),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.last_layer = nn.TransformerEncoderLayer(
            d_model=n_output, nhead=8, batch_first=True
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last_layer(out)
        return out


model = ConvNet1D(1, 8, 16, 64)
print(model)
# data = torch.randn(batch_size, 1, 1250)
# out = model(data)
# print(out.size())
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 学習をするフェーズで初めて必要
