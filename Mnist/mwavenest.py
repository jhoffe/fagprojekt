import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False, num_workers=2)

def batch_to_tensor(batch, device='cuda'):
    if isinstance(batch, list):
        return [batch_to_tensor(x, device=device) for x in batch]
    else:
        return torch.from_numpy(batch).to(device)

def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


class CausalModel(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=28 * 28, layers=3):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers

        super(CausalModel, self).__init__()

        self.flatten = nn.Flatten()
        self.causal_conv = CausalConv1d(in_channels=input_size, out_channels=output_size, kernel_size=8)
        self.conv_stack = nn.Sequential(
            self.causal_conv(),
            nn.ReLU(),
            nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, batch):
        x, target = batch
        x = self.flatten(x)
        pred = self.conv_stack(x)
        loss = nn.MSELoss(reduction='mean')
        output = loss(pred, target)
        return output

model = CausalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
running_loss = np.inf
val_loss = np.inf


for epoch in range(10):
    model.train()
    for batch in enumerate(training_loader):
        loss = model.forward(batch)
        running_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.train(False)
    for batch in enumerate(validation_loader):
        loss = model.forward(batch)
        val_loss.append(loss)