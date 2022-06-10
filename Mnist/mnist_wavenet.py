import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Normal
from torch.utils.data import DataLoader

warnings.simplefilter("ignore")


def show_img(img):
    plt.figure()
    plt.imshow(img.detach().cpu().numpy().reshape((28, 28)))
    plt.show()


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_mix_gaussian(y, log_scale_min=-7.0):
    """
    Sample from (discretized) mixture of gaussian distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    C = y.size(1)
    if C == 2:
        nr_mix = 1
    else:
        assert y.size(1) % 3 == 0
        nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)

    if C == 2:
        logit_probs = None
    else:
        logit_probs = y[:, :, :nr_mix]

    if nr_mix > 1:
        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, nr_mix)

        # Select means and log scales
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    else:
        if C == 2:
            means, log_scales = y[:, :, 0], y[:, :, 1]
        elif C == 3:
            means, log_scales = y[:, :, 1], y[:, :, 2]
        else:
            assert False, "shouldn't happen"

    scales = torch.exp(log_scales)
    dist = Normal(loc=means, scale=scales)
    x = dist.sample()

    x = torch.clamp(x, min=-1.0, max=1.0)
    return x


class Conv1d(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, kw + (kw - 1) * (dilation - 1), input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                # fairseq.modules.conv_tbc.ConvTBC
                weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None


class CausalConv1d(Conv1d):
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


def receptive_field_size(total_layers, num_cycles, kernel_size,
                         dilation=lambda x: 1):
    """Compute receptive field size
    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.
    Returns:
        int: receptive field size in sample
    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


class CausalModel(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=28 * 28, kernel_size=3, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size

        super(CausalModel, self).__init__()

        ch = 256

        self.stacks = 2
        self.layers = 4
        assert self.layers % self.stacks == 0
        layers_per_stack = self.layers // self.stacks

        self.start_conv = Conv1d(in_channels=1, out_channels=ch, kernel_size=1)
        self.conv_layers = nn.ModuleList()

        for layer in range(self.layers):
            # dilation = 2**(layer % layers_per_stack)
            dilation = 1

            conv = CausalConv1d(in_channels=ch, out_channels=ch, kernel_size=kernel_size, bias=True, dilation=dilation)
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            Conv1d(in_channels=ch, out_channels=ch, kernel_size=1),
            nn.ReLU(),
            Conv1d(in_channels=ch, out_channels=2, kernel_size=1),
            nn.Sigmoid()
        ])

        self.receptive_field = receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def forward(self, x):
        x = self.start_conv(x)
        for f in self.conv_layers:
            x = f(x)

        for f in self.last_conv_layers:
            x = f(x)

        return x

    def incremental_forward(self, T=784):
        self.clear_buffer()
        self.eval()

        B = 1

        init_input = torch.zeros(B, 1, self.receptive_field).cuda()
        generated = init_input

        for t in range(T):
            # print(generated[:, :, t:t + self.receptive_field - 1], generated[:, :, t:t + self.receptive_field - 1].size())
            probdist = F.softmax(self.forward(generated[:, :, t:t + self.receptive_field - 1]), dim=1)[:, :,
                       -1].squeeze().detach().cpu().numpy()
            output = torch.tensor([[[np.random.choice([0, 1], p=probdist)]]], device='cuda')
            generated = torch.cat((generated, output), 2)

        self.clear_buffer()
        return generated[:, :, self.receptive_field:]

    def clear_buffer(self):
        self.start_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass


SEED = 42
TRAIN_UPDATES = 30000
BATCH_SIZE = 32
LR = 3e-3
DEVICE = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
CPU_CORES = int(os.environ["CPU_CORES"]) if os.getenv("CPU_CORES") is not None else 4

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1),
    transforms.Lambda(lambda x: x.flatten(start_dim=1))
])
target_transform = transforms.Compose([
    transforms.Lambda(lambda x: x > 0.5),
    transforms.Lambda(lambda x: x.type(torch.LongTensor).squeeze())
])

train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, transform=default_transform,
                                           download=True)
val_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, transform=default_transform,
                                         download=True)
train_dataloader = DataLoader(train_dataset, num_workers=CPU_CORES, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = CausalModel(kernel_size=3).cuda()
model.load_state_dict(torch.load("Mnist/model.pt"))
# loss_fn = CrossEntropyLoss()
# optimizer = Adam(params=model.parameters(), lr=LR)
#
# for epoch in range(50):
#     print(f"Epoch: {epoch + 1}")
#     pbar = tqdm(train_dataloader)
#
#     mean_loss = 0
#
#     for x, _ in pbar:
#         y = target_transform(x).cuda()
#         yh = model.forward(x.cuda())
#         log_probs = F.log_softmax(yh)
#
#         loss = loss_fn(yh, y)
#         optimizer.zero_grad()
#         loss.backward()
#
#         optimizer.step()
#         pbar.set_description(desc=f"NLL={loss}")
#         mean_loss += loss
#     mean_loss = mean_loss / len(pbar)
#     print(mean_loss)
#
# torch.save(model.state_dict(), "Mnist/model.pt")
