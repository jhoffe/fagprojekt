import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataPreprocessor import  *

MnistInstance = MnistDataset()
trainingSet = MnistInstance.TrainingSet
testset = MnistInstance.TestingSet
batch_size = 16

trainloader = torch.utils.data.DataLoader(trainingSet, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

### Alternative causal conv function
# def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
#   pad = (kernel_size - 1) * dilation
#   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
numEpochs = 10

### IMPORTANT NOTE: In order to make the MSE loss function work, we need to assign the correct labels to the dataloader
### This is not the case at the moment.
def main():
    for epoch in range(numEpochs):  # loop over the dataset multiple times

        optimizer.zero_grad()
        running_loss = 0.0
        for i, batch in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    main()

