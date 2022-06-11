import os

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import NLLLoss
from torch.optim import Adam
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

'''
Skal nok slettes.
'''


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

        self.causal_conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=bias)

    def forward(self, x):




if __name__ == "__main__":
    SEED = 42
    TRAIN_UPDATES = 30000
    BATCH_SIZE = 32
    LR = 6e-4
    DEVICE = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    CPU_CORES = int(os.environ['CPU_CORES'])

    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0, 1),
        transforms.Lambda(lambda x: x.flatten(start_dim=1))
    ])
    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: x > 0.5),
        transforms.Lambda(lambda x: x.float())
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, transform=default_transform,
                                               download=True)
    val_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, transform=default_transform,
                                             download=True)
    train_dataloader = DataLoader(train_dataset, num_workers=CPU_CORES, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # model = CausalModel().to(device=DEVICE)
    # optimizer = Adam(lr=LR, params=model.parameters())
    # loss_fn = NLLLoss()
    #
    # pbar = tqdm(train_dataloader)
    #
    # for (batch, _) in pbar:
    #     x = batch
    #
    #     yh = model.forward(x)
    #     loss = loss_fn(y, yh)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     pbar.set_description(desc=f"Training: MSE={loss}")