import torch
from torch import nn

from layers import LinearGaussian, ReluGaussian, MeanFieldConv2d, \
    AveragePoolGaussian, ReluVDO


class LinearDVI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = LinearGaussian(28 * 28, 300, certain=True)
        self.fc2 = ReluGaussian(300, 100) if not args.var_network else ReluVDO(
            300, 100, use_det=not args.mcvi)
        self.fc3 = ReluGaussian(100, 10)

        if args.mcvi:
            self.mcvi()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

    def mean_forward(self, x):
        x = self.fc1.mean_forward(x)
        x = self.fc2.mean_forward(x)
        return self.fc3.mean_forward(x)

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
        super().__init__()

        self.conv1 = MeanFieldConv2d(1, 20, 5, padding=2, certain=True)
        self.conv2 = MeanFieldConv2d(20, 50, 5)
        self.fc1 = ReluGaussian(800, 500)
        self.fc2 = ReluGaussian(500, 100)
        self.fc3 = ReluGaussian(100, 10)

        self.avg_pool = AveragePoolGaussian(kernel_size=(2, 2))

        if args.mcvi:
            self.mcvi()

    def forward(self, x):
        x = self.avg_pool(self.conv1(x))
        x = self.avg_pool(self.conv2(x))

        x_mean = x[0]
        x_var = x[1]

        x_mean = x_mean.view(-1, 400)
        if x_var is not None:
            x_var = x_var.view(-1, 400)
            x_var = torch.diag_embed(x_var)

        x = (x_mean, x_var)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def determenistic(self):
        for module in self.children():
            if hasattr(module, "determenistic"):
                module.determenistic()

    def mcvi(self):
        for module in self.children():
            if hasattr(module, "mcvi"):
                module.mcvi()

    def mean_forward(self, x):
        x = self.avg_pool(self.conv1.mean_forward(x))
        x = self.avg_pool(self.conv2.mean_forward(x))
        x = x.view(-1, 10)
        x = self.fc1.mean_forward(x)
        x = self.fc2.mean_forward(x)
        x = self.fc3.mean_forward(x)
        return x
