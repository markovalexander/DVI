import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Independent, Normal

from bayesian_utils import kl_gaussian, softrelu, matrix_diag_part, kl_loguni, \
    compute_linear_var, compute_relu_var, standard_gaussian, gaussian_cdf

EPS = 1e-6


class LinearGaussian(nn.Module):
    def __init__(self, in_features, out_features, certain=False,
                 deterministic=True):
        """
        Applies linear transformation y = xA^T + b

        A and b are Gaussian random variables

        :param in_features: input dimension
        :param out_features: output dimension
        :param certain:  if true, than x is equal to its mean and has no variance
        """

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.W_logvar = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        self._initialize_weights()
        self._construct_priors()

        self.certain = certain
        self.deterministic = deterministic
        self.mean_forward = False
        self.zero_mean = False

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.W)
        nn.init.normal_(self.bias)

        nn.init.uniform_(self.W_logvar, a=-10, b=-7)
        nn.init.uniform_(self.bias_logvar, a=-10, b=-7)

    def _construct_priors(self):
        self.W_mean_prior = nn.Parameter(torch.zeros_like(self.W),
                                         requires_grad=False)
        self.W_var_prior = nn.Parameter(torch.ones_like(self.W_logvar) * 0.1,
                                requires_grad=False)

        self.bias_mean_prior = nn.Parameter(torch.zeros_like(self.bias),
                                            requires_grad=False)
        self.bias_var_prior = nn.Parameter(
            torch.ones_like(self.bias_logvar) * 0.1,
                                requires_grad=False)

    def compute_kl(self):
        weights_kl = kl_gaussian(self.W, torch.exp(self.W_logvar),
                                 self.W_mean_prior, self.W_var_prior)
        bias_kl = kl_gaussian(self.bias, torch.exp(self.bias_logvar),
                              self.bias_mean_prior, self.bias_var_prior)
        return weights_kl + bias_kl

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        """
        Compute expectation and variance after linear transform
        y = xA^T + b

        :param x: input, size [batch, in_features]
        :return: tuple (y_mean, y_var) for deterministic mode:,  shapes:
                 y_mean: [batch, out_features]
                 y_var:  [batch, out_features, out_features]

                 tuple (sample, None) for MCVI mode,
                 sample : [batch, out_features] - local reparametrization of output
        """
        x = self._apply_activation(x)
        if self.zero_mean:
            return self._zero_mean_forward(x)
        elif self.mean_forward:
            return self._mean_forward(x)
        elif self.deterministic:
            return self._det_forward(x)
        else:
            return self._mcvi_forward(x)

    def _mcvi_forward(self, x):
        W_var = torch.exp(self.W_logvar)
        bias_var = torch.exp(self.bias_logvar)

        if self.certain:
            x_mean = x
            x_var = None
        else:
            x_mean = x[0]
            x_var = x[1]

        y_mean = F.linear(x_mean, self.W.t()) + self.bias

        if self.certain or not self.deterministic:
            xx = x_mean * x_mean
            y_var = torch.diag_embed(F.linear(xx, W_var.t()) + bias_var)
        else:
            y_var = compute_linear_var(x_mean, x_var, self.W, W_var, self.bias,
                                       bias_var)

        dst = MultivariateNormal(loc=y_mean, covariance_matrix=y_var)
        sample = dst.rsample()
        return sample, None

    def _det_forward(self, x):
        W_var = torch.exp(self.W_logvar)
        bias_var = torch.exp(self.bias_logvar)

        if self.certain:
            x_mean = x
            x_var = None
        else:
            x_mean = x[0]
            x_var = x[1]

        y_mean = F.linear(x_mean, self.W.t()) + self.bias

        if self.certain:
            xx = x_mean * x_mean
            y_var = torch.diag_embed(F.linear(xx, W_var.t()) + bias_var)
        else:
            y_var = compute_linear_var(x_mean, x_var, self.W, W_var, self.bias,
                                       bias_var)

        return y_mean, y_var

    def _mean_forward(self, x):
        if not isinstance(x, tuple):
            x_mean = x
        else:
            x_mean = x[0]

        y_mean = F.linear(x_mean, self.W.t()) + self.bias
        return y_mean, None

    def _zero_mean_forward(self, x):
        if not isinstance(x, tuple):
            x_mean = x
            x_var = None
        else:
            x_mean = x[0]
            x_var = x[1]

        y_mean = F.linear(x_mean, torch.zeros_like(self.W).t()) + self.bias

        W_var = torch.exp(self.W_logvar)
        bias_var = torch.exp(self.bias_logvar)

        if x_var is None:
            xx = x_mean * x_mean
            y_var = torch.diag_embed(F.linear(xx, W_var.t()) + bias_var)
        else:
            y_var = compute_linear_var(x_mean, x_var, torch.zeros_like(self.W),
                                       W_var, self.bias, bias_var)

        if self.deterministic:
            return y_mean, y_var
        else:
            dst = MultivariateNormal(loc=y_mean, covariance_matrix=y_var)
            sample = dst.rsample()
            return sample, None

    def _apply_activation(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


# TODO: разобраться почему так долго стало работать
class ReluGaussian(LinearGaussian):
    def _apply_activation(self, x):
        print('applying relu activation...')
        x_mean = x[0]
        x_var = x[1]

        if not self.deterministic:
            z_mean = F.relu(x_mean)
            z_var = None
        else:
            x_var_diag = matrix_diag_part(x_var)
            sqrt_x_var_diag = torch.sqrt(x_var_diag + EPS)
            mu = x_mean / (sqrt_x_var_diag + EPS)

            z_mean = sqrt_x_var_diag * softrelu(mu)
            z_var = compute_relu_var(x_var, x_var_diag, mu)
        print('finished')
        return z_mean, z_var


class LinearVDO(LinearGaussian):
    def __init__(self, in_features, out_features,
                 alpha_shape=(1, 1), certain=False, deterministic=True):
        super(LinearVDO, self).__init__(in_features, out_features, certain,
                                        deterministic)
        self.alpha_shape = alpha_shape
        self.log_alpha = nn.Parameter(torch.Tensor(*alpha_shape))
        self.log_alpha.data.fill_(-5.0)
        self.zero_mean = False

    def compute_kl(self):
        return self.W.nelement() * kl_loguni(
            self.log_alpha) / self.log_alpha.nelement()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', alpha_shape=' + str(self.alpha_shape) + ')'


class ReluVDO(LinearVDO):
    def _apply_activation(self, x):
        print('applying linear activation...')
        x_mean = x[0]
        x_var = x[1]

        if not self.deterministic:
            z_mean = F.relu(x_mean)
            z_var = None
        else:
            x_var_diag = matrix_diag_part(x_var)
            sqrt_x_var_diag = torch.sqrt(x_var_diag + EPS)
            mu = x_mean / (sqrt_x_var_diag + EPS)

            z_mean = sqrt_x_var_diag * softrelu(mu)
            z_var = compute_relu_var(x_var, x_var_diag, mu)

        print('finish!')
        return z_mean, z_var


class DetermenisticReluLinear(ReluGaussian):
    def __init__(self, in_features, out_features, certain=False,
                 deterministic=True):
        """
        Applies linear transformation y = xA^T + b

        A and b are Gaussian random variables

        :param in_features: input dimension
        :param out_features: output dimension
        :param certain:  if true, than x is equal to its mean and has no variance
        """

        super().__init__(in_features, out_features, certain, deterministic)
        self.W_logvar.requires_grad = False
        self.bias_logvar.requires_grad = False

    def compute_kl(self):
        return 0


class MeanFieldConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 activation='relu', padding=0, certain=False,
                 deterministic=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.activation = activation.strip().lower()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        self.W = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_logvar = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_channels))

        self._initialize_weights()
        self._construct_priors()

        self.certain = certain
        self.deterministic = deterministic
        self.mean_forward = False
        self.zero_mean = False

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.W)
        nn.init.normal_(self.bias)

        nn.init.uniform_(self.W_logvar, a=-10, b=-7)
        nn.init.uniform_(self.bias_logvar, a=-10, b=-7)

    def _construct_priors(self):
        self.W_mean_prior = nn.Parameter(torch.zeros_like(self.W),
                                         requires_grad=False)
        self.W_var_prior = nn.Parameter(torch.ones_like(self.W_logvar) * 0.1,
                                requires_grad=False)

        self.bias_mean_prior = nn.Parameter(torch.zeros_like(self.bias),
                                            requires_grad=False)
        self.bias_var_prior = nn.Parameter(
            torch.ones_like(self.bias_logvar) * 0.1,
                                requires_grad=False)

    def compute_kl(self):
        weights_kl = kl_gaussian(self.W, torch.exp(self.W_logvar),
                                 self.W_mean_prior, self.W_var_prior)
        bias_kl = kl_gaussian(self.bias, torch.exp(self.bias_logvar),
                              self.bias_mean_prior, self.bias_var_prior)
        return weights_kl + bias_kl

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        x = self._apply_activation(x)
        if self.zero_mean:
            return self._zero_mean_forward(x)
        elif self.mean_forward:
            return self._mean_forward(x)
        elif self.deterministic:
            return self._det_forward(x)
        else:
            return self._mcvi_forward(x)

    def _zero_mean_forward(self, x):
        if self.certain or not self.deterministic:
            x_mean = x if not isinstance(x, tuple) else x[0]
            x_var = x_mean * x_mean
        else:
            x_mean = x[0]
            x_var = x[1]

        W_var = torch.exp(self.W_logvar)
        bias_var = torch.exp(self.bias_logvar)

        z_mean = F.conv2d(x_mean, torch.zeros_like(self.W), self.bias,
                          self.stride,
                          self.padding)
        z_var = F.conv2d(x_var, W_var, bias_var, self.stride,
                         self.padding)

        if self.deterministic:
            return z_mean, z_var
        else:
            dst = Independent(Normal(z_mean, z_var), 1)
            sample = dst.rsample()
            return sample, None

    def _mean_forward(self, x):
        if not isinstance(x, tuple):
            x_mean = x
        else:
            x_mean = x[0]

        z_mean = F.conv2d(x_mean, self.W, self.bias,
                          self.stride,
                          self.padding)
        return z_mean, None

    def _det_forward(self, x):
        if self.certain and isinstance(x, tuple):
            x_mean = x[0]
            x_var = x_mean * x_mean
        elif not self.certain:
            x_mean = x[0]
            x_var = x[1]
        else:
            x_mean = x
            x_var = x_mean * x_mean

        W_var = torch.exp(self.W_logvar)
        bias_var = torch.exp(self.bias_logvar)

        z_mean = F.conv2d(x_mean, self.W, self.bias,
                          self.stride,
                          self.padding)
        z_var = F.conv2d(x_var, W_var, bias_var, self.stride,
                         self.padding)
        return z_mean, z_var

    def _mcvi_forward(self, x):
        if self.certain or not self.deterministic:
            x_mean = x if not isinstance(x, tuple) else x[0]
            x_var = x_mean * x_mean
        else:
            x_mean = x[0]
            x_var = x[1]

        W_var = torch.exp(self.W_logvar)
        bias_var = torch.exp(self.bias_logvar)

        z_mean = F.conv2d(x_mean, self.W, self.bias,
                          self.stride,
                          self.padding)
        z_var = F.conv2d(x_var, W_var, bias_var, self.stride,
                         self.padding)

        dst = Independent(Normal(z_mean, z_var), 1)
        sample = dst.rsample()
        return sample, None

    def _apply_activation(self, x):
        print('applying field activation...')
        if self.activation == 'relu' and not self.certain:
            x_mean, x_var = x
            if x_var is None:
                x_var = x_mean * x_mean

            sqrt_x_var = torch.sqrt(x_var + EPS)
            mu = x_mean / sqrt_x_var
            z_mean = sqrt_x_var * softrelu(mu)
            z_var = x_var * (mu * standard_gaussian(mu) + (
                    1 + mu ** 2) * gaussian_cdf(mu))
        elif not self.certain:
            z_mean, z_var = x
        else:
            print('finish')
            return x
        print('finish!')
        return z_mean, z_var

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', activation=' + str(self.activation) + ')'


class AveragePoolGaussian(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if not isinstance(x, tuple):
            raise ValueError(
                "Input for pooling layer should be tuple of tensors")

        x_mean, x_var = x
        z_mean = F.avg_pool2d(x_mean, self.kernel_size, self.stride,
                              self.padding)
        if x_var is None:
            z_var = None
        else:
            n = self.kernel_size[0] * self.kernel_size[1]
            z_var = F.avg_pool2d(x_var, self.kernel_size, self.stride,
                                 self.padding) / n
        return z_mean, z_var

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'kernel_size= ' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) + ')'
