import torch
from torch import nn

from layers import LinearGaussian, ReluGaussian, MeanFieldConv2d, \
    AveragePoolGaussian, ReluVDO, DetermenisticReluLinear


class LinearDVI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = LinearGaussian(784, 300, certain=True)
        self.fc2 = ReluGaussian(300, 100)
        self.fc3 = ReluGaussian(100, 10)

        if args.mcvi:
            self.set_flag('deterministic', False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

    def set_flag(self, flag_name, value):
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)


class LinearVDO(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = LinearGaussian(784, 300, certain=True)
        self.fc2 = ReluVDO(300, 100)

        if args.n_var_layers > 1:
            self.fc3 = ReluVDO(100, 10)
        else:
            self.fc3 = DetermenisticReluLinear(100, 10)

        if args.mcvi:
            self.set_flag('deterministic', False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

    def set_flag(self, flag_name, value):
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def zero_mean(self, mode=True):
        for layer in self.children():
            if isinstance(layer, ReluVDO):
                layer.set_flag('zero_mean', mode)

    def print_alphas(self):
        i = 1
        for layer in self.children():
            if hasattr(layer, 'log_alpha'):
                print('{} var_layer log_alpha={:.5f}'.format(i,
                                                             layer.log_alpha.item()))
                i += 1


class LeNetDVI(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = MeanFieldConv2d(1, 6, 5, padding=2, certain=True)
        self.conv2 = MeanFieldConv2d(6, 16, 5)
        self.fc1 = ReluGaussian(16 * 5 * 5, 120)
        self.fc2 = ReluGaussian(120, 84)
        self.fc3 = ReluGaussian(84, 10)

        self.avg_pool = AveragePoolGaussian(kernel_size=(2, 2))

        if args.mcvi:
            self.set_flag('deterministic', False)

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

    def set_flag(self, flag_name, value):
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)


class LeNetVDO(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = MeanFieldConv2d(1, 6, 5, padding=2, certain=True)
        self.conv2 = MeanFieldConv2d(6, 16, 5)
        self.fc1 = ReluVDO(16 * 5 * 5, 120)

        if args.n_var_layers > 1:
            self.fc2 = ReluVDO(120, 84)
        else:
            self.fc2 = DetermenisticReluLinear(120, 84)

        if args.n_var_layers > 2:
            self.fc3 = ReluVDO(84, 10)
        else:
            self.fc3 = DetermenisticReluLinear(84, 10)

        self.avg_pool = AveragePoolGaussian(kernel_size=(2, 2))

        if args.mcvi:
            self.set_flag('deterministic', False)

    def zero_mean(self, mode=True):
        for layer in self.children():
            if isinstance(layer, ReluVDO):
                layer.set_flag('zero_mean', mode)

    def forward(self, x):
        x = self.avg_pool(self.conv1(x))
        x = self.avg_pool(self.conv2(x))

        x_mean = x[0]
        x_var = x[1]

        x_mean = x_mean.view(-1, 1250)
        if x_var is not None:
            x_var = x_var.view(-1, 1250)
            x_var = torch.diag_embed(x_var)

        x = (x_mean, x_var)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def set_flag(self, flag_name, value):
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def print_alphas(self):
        i = 1
        for layer in self.children():
            if hasattr(layer, 'log_alpha'):
                print('{} var_layer log_alpha={:.5f}'.format(i,
                                                             layer.log_alpha.item()))
                i += 1
