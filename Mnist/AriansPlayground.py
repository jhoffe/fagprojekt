import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataPreprocessor import  *

MnistInstance = MnistDataset()
trainingSet = MnistInstance.TrainingSet
testset = MnistInstance.TestingSet
batch_size = 4

trainloader = torch.utils.data.DataLoader(trainingSet, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


class CausalModel(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=28 * 28, layers=3):
        super(CausalModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers

        self.flatten = nn.Flatten()
        self.causal_conv = CausalConv1d(in_channels=input_size, out_channels=output_size, kernel_size=8)
        self.end_conv1d = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1, bias=True)

        def forward(self, x):
            x = self.flatten(x)
            for _ in range(self.layers):
                x = F.relu(self.causal_conv.forward(x))
            x = F.relu(self.end_conv1d.forward(x))
            return x

model = CausalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
running_loss = np.inf
val_loss = np.inf

for epoch in range(10):
    model.train()
    for batch in enumerate(trainloader):
        loss = model.forward(batch)
        running_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.train(False)
    for batch in enumerate(testloader):
        loss = model.forward(batch)
        val_loss.append(loss)
