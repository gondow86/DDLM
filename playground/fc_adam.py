from random import shuffle
from tkinter import image_names
from turtle import down
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# dataset
train_dataset = datasets.MNIST(
    root="../data/", train=True, transform=transform, download=True
)

test_dataset = datasets.MNIST(root="../data/", train=False, transform=transform)

# Data Loader
batch_size = 256
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# Model
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


criterion = nn.CrossEntropyLoss()

adam = ["Adam"]
loss_fc = {}
acc_fc = {}
optimizer = {}

for key in adam:
    loss_fc[key] = []
    acc_fc[key] = []

lr = 0.01

for key in adam:
    model = Net()

    optimizer["Adam"] = optim.Adam(model.parameters(), lr)

    for epoch in range(20):
        model.train()

    for image, label in train_loader:
        image = image.view(-1, 28 * 28)
        optimizer[key].zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer[key].step()
        loss_fc[key].append(loss.item())

    model.eval()
    with torch.no_grad():
        for t_image, t_label in test_loader:
            t_image = t_image.view(-1, 28 * 28)
            output = model(t_image)
            _, predicted = torch.max(output, 1)
            class_correct = (predicted == t_label).sum().item()
            acc = class_correct / len(predicted) * 100
            acc_fc[key].append(acc)
