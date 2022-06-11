import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from psutil import cpu_count
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, A=False, **kwargs):
        super(CausalConv1d, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A
        self.padding = (kernel_size - 1) * dilation + A * 1

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation,
                                **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        conv_out = self.conv1d(x)
        if self.A:
            return conv_out[:, :, :-1]
        else:
            return conv_out


class WaveNIST(pl.LightningModule):
    def __init__(self, layers=3, hidden=256, kernel_size=3, output_classes=256):
        super(WaveNIST, self).__init__()
        self.output_classes = output_classes

        self.first_conv = CausalConv1d(in_channels=1, out_channels=hidden, dilation=1, kernel_size=kernel_size, A=True,
                                       bias=True)
        self.hidden_convs = nn.ModuleList([CausalConv1d(in_channels=hidden, out_channels=hidden, dilation=1,
                                                        kernel_size=kernel_size, A=False, bias=True)] * layers)
        self.end_conv = CausalConv1d(in_channels=hidden, out_channels=output_classes, dilation=1,
                                     kernel_size=kernel_size, A=False, bias=True)

        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()
        self.loss_fn = nn.NLLLoss()

    def forward(self, x):
        x = self.first_conv(x)

        for h in self.hidden_convs:
            x = self.activation(h(x))

        x = self.end_conv(x)

        return torch.log_softmax(x, 1)

    def generate(self, batch_size, output_length):
        generated = torch.zeros((batch_size, output_length))

        for t in range(output_length):
            p = self.forward(generated)
            y = torch.multinomial(p[:, t, :], num_samples=1)
            generated[:, t] = y[:, 0]

        return generated

    def discretize_input(self, x):
        return torch.bucketize(x, torch.tensor(
            [1 / self.output_classes * i for i in range(self.output_classes - 1)], device=x.get_device())).squeeze()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        y = self.discretize_input(x)
        p = self.forward(x)
        loss = self.loss_fn(p, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        y = self.discretize_input(x)
        p = self.forward(x)
        val_loss = self.loss_fn(p, y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


classes = 256
input_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten(1))])

train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=input_transforms)
val_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=input_transforms)

train_loader = DataLoader(train_set, batch_size=32, num_workers=cpu_count())
val_loader = DataLoader(val_set, batch_size=32, num_workers=cpu_count())

pl.seed_everything(42, workers=True)

logger = WandbLogger(project="wavenist")

model = WaveNIST()
trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                     devices=-1 if torch.cuda.is_available() else None, max_epochs=1000,
                     logger=logger,
                     default_root_dir="models/")
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
