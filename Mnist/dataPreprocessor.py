import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


cwd = os.getcwd()
MnistDataset_training = torchvision.datasets.MNIST(root="{}\Data".format(os.getcwd()), download=True, train=True, transform=ToTensor())
MnistDataset_testing = torchvision.datasets.MNIST(root="{}\Data".format(os.getcwd()), download=True, train=False, transform=ToTensor())


class MnistDataset(Dataset):
    def __init__(self, transform=ToTensor()):
        # if(os.path.isdir(img_dir)):
        #    os.mkdir(img_dir)
        self.TrainingSet = torchvision.datasets.MNIST(root="{}\Data".format(os.getcwd()), download=True,
                                                           train=True, transform=transform)
        self.TestingSet = torchvision.datasets.MNIST(root="{}\Data".format(os.getcwd()), download=True,
                                                          train=False, transform=transform)
    def __len__(self):
        return len(self.TrainingSet), len(self.TestingSet)

    def __getitem__(self, idx, trainingSet=True):
        image, label = self.TrainingSet[0], self.TrainingSet[1] if(trainingSet) else self.TestingSet[0], self.TestingSet[1]
        return image, label

    def plotExample(self):
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        labels_map = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(MnistDataset_training), size=(1,)).item()
            img, label = MnistDataset_training[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()

