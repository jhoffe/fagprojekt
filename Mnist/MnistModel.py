import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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


class UniformBatchSampler:
    def __init__(self, dataset_size: int, num_steps: int, batch_size: int, seed=None):
        np.random.seed(seed)
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.world = np.arange(self.dataset_size)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = np.random.choice(self.world, self.batch_size, replace=False).tolist()
            yield batch

    def __len__(self) -> int:
        return self.num_steps


if __name__ == "__main__":
    SEED = 42
    TRAIN_UPDATES = 50000
    BATCH_SIZE = 16

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0,), (1,))])

    MnistDataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, transform=transform, download=True)
    batch_sampler = UniformBatchSampler(len(MnistDataset), TRAIN_UPDATES, BATCH_SIZE, seed=SEED)
    data_loader = DataLoader(MnistDataset, batch_sampler=batch_sampler)

    model = CausalModel()
    optimizer = Adam(lr=1e-3, params=model.parameters())
    loss_fn = MSELoss()

    pbar = tqdm(data_loader)

    running_mse = 0
    running_acc = 0

    for batch in pbar:
        y = model.forward(batch)
        loss = loss_fn(batch, y)
        loss.backward()
        optimizer.step()

        pbar.set_description(desc=f"MSE={loss}")
