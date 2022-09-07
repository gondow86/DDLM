from torch import nn
import torch

conv1d = nn.Conv1d(2, 3, 5)
# print(conv1d.weight.shape, conv1d.weight, sep='\n')
# print(conv1d.bias)

x = torch.rand(4, 2, 6)
# print(x)

y = conv1d(x)
print(conv1d.weight[0], y[:, 0], sep="\n")
