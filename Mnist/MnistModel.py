import os

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
import matplotlib.pyplot as plt

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

        self.causal_conv = CausalConv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size,
                                        bias=bias)
        self.end_conv1d = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1, bias=True)

    def forward(self, x):
        for _ in range(self.layers):
            x = F.relu(self.causal_conv.forward(x))

        x = F.relu(self.end_conv1d.forward(x))

        return x.squeeze()


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


def batch_to_tensor(batch, device='cuda'):
    if isinstance(batch, list):
        return [batch_to_tensor(x, device=device) for x in batch]
    else:
        return torch.from_numpy(batch).to(device)


if __name__ == "__main__":
    SEED = 42
    TRAIN_UPDATES = 30000
    BATCH_SIZE = 32
    LR = 6e-4
    DEVICE = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    CPU_CORES = int(os.environ['CPU_CORES'])

    default_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])

    transform_input = transforms.Compose(
        [transforms.RandomErasing(p=1, scale=(0.1, 0.4)),
         transforms.Lambda(lambda x: x.flatten(start_dim=1))])
    transform_output = transforms.Compose([transforms.Lambda(lambda x: x.flatten(start_dim=1))])

    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, transform=default_transform, download=True)
    val_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, transform=default_transform, download=True)
    batch_sampler = UniformBatchSampler(len(train_dataset), TRAIN_UPDATES, BATCH_SIZE, seed=SEED)
    train_dataloader = DataLoader(train_dataset, num_workers=CPU_CORES, batch_sampler=batch_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = CausalModel().to(device=DEVICE)
    optimizer = Adam(lr=LR, params=model.parameters())
    loss_fn = MSELoss()

    pbar = tqdm(train_dataloader)

    for (batch, _) in pbar:
        x = transform_input(batch).to(device=DEVICE)
        y = transform_output(batch).to(device=DEVICE)

        yh = model.forward(x)
        loss = loss_fn(y, yh)
        loss.backward()
        optimizer.step()

        pbar.set_description(desc=f"Training: MSE={loss}")

    model.eval()
    for (batch, _) in val_dataloader:
        x = transform_input(batch).to(device=DEVICE)
        y = transform_output(batch).to(device=DEVICE)
        yh = model.forward(x)

        xt = x[0, :].resize((28, 28))
        yt = y[0, :].resize((28, 28))
        yht = yh[0, :].resize((28, 28))

        fig, axs = plt.subplots(1, 3)

        axs[0].imshow(xt)
        axs[0].title.set_text("Occluded original")
        axs[1].imshow(yt)
        axs[1].title.set_text("Real original")
        axs[2].imshow(yht)
        axs[2].title.set_text("Our guess")

        fig.save("mnist_example.png")

        break
