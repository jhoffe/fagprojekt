import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

MnistDataset = torchvision.datasets.MNIST()

data_loader = torch.utils.data.DataLoader(MnistDataset,
                                          batch_size=4,
                                          shuffle=True)


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)


class CausalModel(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=28 * 28, layers=3, kernel_size=2, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.kernel_size = kernel_size

        super(CausalModel, self).__init__()

        self.causal_conv = CausalConv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, bias=bias)
        self.end_conv1d = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1, bias=True)

    def forward(self, x):
        for _ in range(self.layers):
            x = F.relu(self.causal_conv.forward(x))

        x = F.relu(self.end_conv1d.forward(x))

        return x
