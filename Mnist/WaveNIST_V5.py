import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import BCELoss
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


EPS = 1.e-5


class WaveNIST(pl.LightningModule):
    def __init__(self, hidden=256, kernel_size=17, output_size=784):
        super(WaveNIST, self).__init__()
        self.output_size = output_size

        self.loss_fn = BCELoss()
        self.net = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=hidden, dilation=1, kernel_size=kernel_size, A=True, bias=True),
            nn.ReLU(),
            CausalConv1d(in_channels=hidden, out_channels=hidden, dilation=2, kernel_size=kernel_size, A=True,
                         bias=True),
            nn.ReLU(),
            CausalConv1d(in_channels=hidden, out_channels=hidden, dilation=4, kernel_size=kernel_size, A=True,
                         bias=True),
            nn.ReLU(),
            CausalConv1d(in_channels=hidden, out_channels=1, dilation=8, kernel_size=kernel_size, A=True, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def generate(self, batch_size):
        generated = torch.zeros((batch_size, 1, self.output_size), device=self.device)

        for t in range(self.output_size):
            p = self.forward(generated).squeeze()
            generated[:, 0, t] = p[:, t]

        return generated.squeeze()

    @staticmethod
    def plot_generated(generated):
        figs = []
        for i in range(generated.size(0)):
            fig = plt.figure()
            plt.imshow(generated[i, :].detach().cpu().numpy().reshape((28, 28)), cmap="gray")
            figs.append(fig)

        return figs

    def training_step(self, batch, batch_idx):
        x, _ = batch
        p = self.forward(x).squeeze()
        loss = self.loss_fn(p, x.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        p = self.forward(x).squeeze()
        val_loss = self.loss_fn(p, x.squeeze())
        self.log("val_loss", val_loss)

        return val_loss

    def on_validation_end(self):
        self.eval()
        generated = self.generate(32)
        figs = self.plot_generated(generated)
        self.logger.log_image(key="digits", images=figs)
        for fig in figs:
             plt.close(fig)
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


input_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten(1))])

train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=input_transforms)
val_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=input_transforms)

train_loader = DataLoader(train_set, batch_size=64, num_workers=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, num_workers=4)

pl.seed_everything(42, workers=True)

logger = WandbLogger(project="wavenist")

model = WaveNIST(hidden=256, kernel_size=7)

trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                     devices=-1 if torch.cuda.is_available() else None, max_epochs=100,
                     logger=logger,
                     default_root_dir="models/")

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
