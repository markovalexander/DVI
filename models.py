import torch
from torch import nn

from layers import LinearGaussian, ReluGaussian


class LinearDVI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = LinearGaussian(28 * 28, 300, certain=True)
        self.fc2 = ReluGaussian(300, 100)
        self.fc3 = ReluGaussian(100, 10)

        if args.mcvi:
            self.mcvi()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

    def mcvi(self):
        self.fc1.mcvi()
        self.fc2.mcvi()
        self.fc3.mcvi()

    def determenistic(self):
        self.fc1.determenistic()
        self.fc2.determenistic()
        self.fc3.determenistic()


class LeNetDVI(nn.Module):
    def __init__(self, args):
        raise NotImplementedError("not implemented yet")
