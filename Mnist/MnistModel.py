import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


MnistDataset = torchvision.datasets.MNIST()

data_loader = torch.utils.data.DataLoader(MnistDataset,
                                          batch_size=4,
                                          shuffle=True)


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

class MnistModel(nn.Module):
    def __init(self, input_size=28*28, output_size=28*28):
        super.__init__(self)
        self.input_size = input_size

        self.conv1 = CausalConv1d(in_channels=input_size, out_channels=output_size, kernel_size=3, dilation=1)

def forward(self, x):

   x = self.conv1(x)
   x = x[:, :, :-self.conv1.padding[0]]

   return x


