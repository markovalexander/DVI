from layers import *


class LinearDVI(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.nonlinearity == 'relu':
            layer_factory = ReluGaussian
        else:
            layer_factory = HeavisideGaussian

        self.fc1 = LinearGaussian(784, 300, certain=True)
        self.fc2 = layer_factory(300, 100)
        self.fc3 = layer_factory(100, 10)

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


class LinearVariance(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = LinearGaussian(784, 300, certain=True)

        if args.nonlinearity == 'relu':
            layer_factory = VarianceReluGaussian
        else:
            layer_factory = VarianceHeavisideGaussian

        self.fc2 = layer_factory(300, 100)

        if args.n_layers > 1:
            self.fc3 = layer_factory(100, 10)
        else:
            self.fc3 = DetermenisticReluGaussian(100, 10)

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

        if args.nonlinearity == 'relu':
            layer_factory = ReluVDO
        else:
            layer_factory = HeavisideVDO

        self.fc2 = layer_factory(300, 100, deterministic=not args.mcvi)

        if args.n_layers > 1:
            self.fc3 = layer_factory(100, 10, deterministic=not args.mcvi)
        else:
            self.fc3 = DetermenisticReluGaussian(100, 10)

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

        if args.nonlinearity == 'relu':
            layer_factory = ReluGaussian
        else:
            layer_factory = HeavisideGaussian

        self.fc1 = layer_factory(16 * 5 * 5, 120)
        self.fc2 = layer_factory(120, 84)
        self.fc3 = layer_factory(84, 10)

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

        if args.nonlinearity == 'relu':
            layer_factory = ReluVDO
        else:
            layer_factory = HeavisideVDO

        self.fc1 = layer_factory(16 * 5 * 5, 120, deterministic=not args.mcvi)

        if args.n_layers > 1:
            self.fc2 = layer_factory(120, 84, deterministic=not args.mcvi)
        else:
            self.fc2 = DetermenisticReluGaussian(120, 84)

        if args.n_layers > 2:
            self.fc3 = layer_factory(84, 10, deterministic=not args.mcvi)
        else:
            self.fc3 = DetermenisticReluGaussian(84, 10)

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


class LeNetVariance(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = MeanFieldConv2d(1, 6, 5, padding=2, certain=True)
        self.conv2 = MeanFieldConv2d(6, 16,
                                     5) if not args.var_conv else VarianceMeanFieldConv2d(
            6, 16, 5)

        if args.nonlinearity == 'relu':
            layer_factory = VarianceReluGaussian
        else:
            layer_factory = VarianceHeavisideGaussian

        self.fc1 = layer_factory(16 * 5 * 5, 120)

        if args.n_layers > 1:
            self.fc2 = layer_factory(120, 84)
        else:
            self.fc2 = DetermenisticReluGaussian(120, 84)

        if args.n_layers > 2:
            self.fc3 = layer_factory(84, 10)
        else:
            self.fc3 = DetermenisticReluGaussian(84, 10)

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
