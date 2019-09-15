from .layers import *


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

        if args.var1:
            self.fc1 = VarianceGaussian(784, 300, certain=True,
                                        sigma_sq=args.use_sqrt_sigma)
        else:
            self.fc1 = DeterministicGaussian(784, 300, certain=True)

        if args.var2:
            self.fc2 = VarianceReluGaussian(300, 100,
                                            sigma_sq=args.use_sqrt_sigma)
        else:
            self.fc2 = DeterministicReluGaussian(300, 100)

        if args.var3:
            self.fc3 = VarianceReluGaussian(100, 10,
                                            sigma_sq=args.use_sqrt_sigma)
        else:
            self.fc3 = DeterministicReluGaussian(100, 10)

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
            self.fc3 = DeterministicReluGaussian(100, 10)

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

        if args.vdo1:
            self.conv1 = MeanFieldConv2dVDO(1, 6, 5, padding=2, certain=True,
                                            deterministic=not args.mcvi,
                                            alpha_shape=(1, 1, 1, 1))
        else:
            self.conv1 = MeanFieldConv2d(1, 6, 5, padding=2, certain=True,
                                         deterministic=not args.mcvi)

        if args.vdo2:
            self.conv2 = MeanFieldConv2dVDO(6, 16, 5,
                                            deterministic=not args.mcvi,
                                            alpha_shape=(1, 1, 1, 1))
        else:
            self.conv2 = MeanFieldConv2d(6, 16, 5, deterministic=not args.mcvi)

        if args.vdo3:
            self.fc1 = ReluVDO(16 * 5 * 5, 120, deterministic=not args.mcvi)
        else:
            self.fc1 = DeterministicReluGaussian(16 * 5 * 5, 120,
                                                 deterministic=not args.mcvi)

        if args.vdo4:
            self.fc2 = ReluVDO(120, 84, deterministic=not args.mcvi)
        else:
            self.fc2 = DeterministicReluGaussian(120, 84,
                                                 deterministic=not args.mcvi)

        if args.vdo5:
            self.fc3 = ReluVDO(84, 10, deterministic=not args.mcvi)
        else:
            self.fc3 = DeterministicReluGaussian(84, 10,
                                                 deterministic=not args.mcvi)

        self.avg_pool = AveragePoolGaussian(kernel_size=(2, 2))

        if args.mcvi:
            self.set_flag('deterministic', False)

    def zero_mean(self, mode=True):
        for layer in self.children():
            if isinstance(layer, ReluVDO) or isinstance(layer,
                                                        MeanFieldConv2dVDO):
                if layer.log_alpha > 3 and mode:
                    layer.set_flag('zero_mean', mode)
                if not mode:
                    layer.set_flag('zero_mean', False)

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

        if args.var1:
            self.conv1 = VarianceMeanFieldConv2d(1, 6, 5, padding=2,
                                                 certain=True,
                                                 deterministic=not args.mcvi,
                                                 sigma_sq=args.use_sqrt_sigma)
        else:
            self.conv1 = MeanFieldConv2d(1, 6, 5, padding=2, certain=True,
                                         deterministic=not args.mcvi)

        if args.var2:
            self.conv2 = VarianceMeanFieldConv2d(6, 16, 5,
                                                 deterministic=not args.mcvi,
                                                 sigma_sq=args.use_sqrt_sigma)
        else:
            self.conv2 = MeanFieldConv2d(6, 16, 5, deterministic=not args.mcvi)

        if args.var3:
            self.fc1 = VarianceReluGaussian(16 * 5 * 5, 120,
                                            deterministic=not args.mcvi,
                                            sigma_sq=args.use_sqrt_sigma)
        else:
            self.fc1 = DeterministicReluGaussian(16 * 5 * 5, 120,
                                                 deterministic=not args.mcvi)

        if args.var4:
            self.fc2 = VarianceReluGaussian(120, 84,
                                            deterministic=not args.mcvi,
                                            sigma_sq=args.use_sqrt_sigma)
        else:
            self.fc2 = DeterministicReluGaussian(120, 84,
                                                 deterministic=not args.mcvi)

        if args.var5:
            self.fc3 = VarianceReluGaussian(84, 10, deterministic=not args.mcvi,
                                            sigma_sq=args.use_sqrt_sigma)
        else:
            self.fc3 = DeterministicReluGaussian(84, 10,
                                                 deterministic=not args.mcvi)

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
