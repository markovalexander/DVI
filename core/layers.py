import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Independent, Normal

from .bayesian_utils import kl_gaussian, softrelu, matrix_diag_part, kl_loguni, \
    compute_linear_var, compute_relu_var, standard_gaussian, gaussian_cdf, \
    compute_heaviside_var

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

    def _get_var(self, param):
        return torch.exp(param)

    def compute_kl(self):
        weights_kl = kl_gaussian(self.W, self._get_var(self.W_logvar),
                                 self.W_mean_prior, self.W_var_prior)
        bias_kl = kl_gaussian(self.bias, self._get_var(self.bias_logvar),
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
        W_var = self._get_var(self.W_logvar)
        bias_var = self._get_var(self.bias_logvar)

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
        W_var = self._get_var(self.W_logvar)
        bias_var = self._get_var(self.bias_logvar)

        if self.certain:
            x_mean = x
            x_var = None
        else:
            x_mean = x[0]
            x_var = x[1]

        y_mean = F.linear(x_mean, self.W.t()) + self.bias

        if self.certain or x_var is None:
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

        W_var = self._get_var(self.W_logvar)
        bias_var = self._get_var(self.bias_logvar)

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


class ReluGaussian(LinearGaussian):
    def _apply_activation(self, x):
        if isinstance(x, tuple):
            x_mean = x[0]
            x_var = x[1]
        else:
            x_mean = x
            x_var = None

        if x_var is None:
            z_mean = F.relu(x_mean)
            z_var = None
        else:
            x_var_diag = matrix_diag_part(x_var)
            sqrt_x_var_diag = torch.sqrt(x_var_diag + EPS)
            mu = x_mean / (sqrt_x_var_diag + EPS)

            z_mean = sqrt_x_var_diag * softrelu(mu)
            z_var = compute_relu_var(x_var, x_var_diag, mu)

        return z_mean, z_var


class HeavisideGaussian(LinearGaussian):
    def _apply_activation(self, x):
        x_mean = x[0]
        x_var = x[1]

        if x_var is None:
            x_var = x_mean * x_mean

        x_var_diag = matrix_diag_part(x_var)

        sqrt_x_var_diag = torch.sqrt(x_var_diag)
        mu = x_mean / (sqrt_x_var_diag + EPS)

        z_mean = gaussian_cdf(mu)
        z_var = compute_heaviside_var(x_var, x_var_diag, mu)

        return z_mean, z_var


class DeterministicGaussian(LinearGaussian):
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


class DeterministicReluGaussian(ReluGaussian):
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


class LinearVDO(nn.Module):

    def __init__(self, in_features, out_features, prior='loguni',
                 alpha_shape=(1, 1), bias=True, deterministic=True):
        super(LinearVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_shape = alpha_shape
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = nn.Parameter(torch.Tensor(*alpha_shape))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.zero_mean = False
        self.permute_sigma = False
        self.prior = prior
        self.kl_fun = kl_loguni
        self.deterministic = deterministic

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        if self.deterministic:
            return self._det_forward(x)
        else:
            return self._mc_forward(x)

    def _mc_forward(self, x):
        if isinstance(x, tuple):
            x_mean = x[0]
            x_var = x[1]
        else:
            x_mean = x

        if self.zero_mean:
            lrt_mean = 0.0
        else:
            lrt_mean = F.linear(x_mean, self.W)
        if self.bias is not None:
            lrt_mean = lrt_mean + self.bias

        sigma2 = torch.exp(self.log_alpha) * self.W * self.W
        if self.permute_sigma:
            sigma2 = sigma2.view(-1)[torch.randperm(
                self.in_features * self.out_features).cuda()].view(
                self.out_features, self.in_features)

        if x_var is None:
            x_var = torch.diag_embed(x_mean * x_mean)

        lrt_cov = compute_linear_var(x_mean, x_var, self.W.t(), sigma2.t())
        dst = MultivariateNormal(lrt_mean, covariance_matrix=lrt_cov)
        return dst.rsample(), None

    def compute_kl(self):
        return self.W.nelement() * self.kl_fun(
            self.log_alpha) / self.log_alpha.nelement()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', alpha_shape=' + str(self.alpha_shape) \
               + ', prior=' + self.prior \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(
            self.bias is not None) + ')'

    def _det_forward(self, x):
        if isinstance(x, tuple):
            x_mean = x[0]
            x_var = x[1]
        else:
            x_mean = x
            x_var = torch.diag_embed(x_mean * x_mean)

        batch_size = x_mean.size(0)
        sigma2 = torch.exp(self.log_alpha) * self.W * self.W
        if self.zero_mean:
            y_mean = torch.zeros(batch_size, self.out_features).to(
                x_mean.device)
        else:
            y_mean = F.linear(x_mean, self.W)
        if self.bias is not None:
            y_mean = y_mean + self.bias

        y_var = compute_linear_var(x_mean, x_var, self.W.t(), sigma2.t())
        return y_mean, y_var

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)


class ReluVDO(LinearVDO):
    def forward(self, x):
        x = self._apply_activation(x)
        return super().forward(x)

    def _apply_activation(self, x):
        if isinstance(x, tuple):
            x_mean = x[0]
            x_var = x[1]
        else:
            x_mean = x
            x_var = None

        if x_var is None:
            z_mean = F.relu(x_mean)
            z_var = None
        else:
            x_var_diag = matrix_diag_part(x_var)
            sqrt_x_var_diag = torch.sqrt(x_var_diag + EPS)
            mu = x_mean / (sqrt_x_var_diag + EPS)

            z_mean = sqrt_x_var_diag * softrelu(mu)
            z_var = compute_relu_var(x_var, x_var_diag, mu)

        return z_mean, z_var


class HeavisideVDO(LinearVDO):
    def forward(self, x):
        x = self._apply_activation(x)
        return super().forward(x)

    def _apply_activation(self, x):
        x_mean = x[0]
        x_var = x[1]

        if x_var is None:
            x_var = x_mean * x_mean

        x_var_diag = matrix_diag_part(x_var)

        sqrt_x_var_diag = torch.sqrt(x_var_diag)
        mu = x_mean / (sqrt_x_var_diag + EPS)

        z_mean = gaussian_cdf(mu)
        z_var = compute_heaviside_var(x_var, x_var_diag, mu)

        return z_mean, z_var


class VarianceGaussian(LinearGaussian):
    def __init__(self, in_features, out_features,
                 certain=False, deterministic=True, sigma_sq=False):
        super().__init__(in_features, out_features, certain, deterministic)
        self.W.data.fill_(0)
        self.W.requires_grad = False
        self.sigma_sq = sigma_sq
        if sigma_sq:
            self.W_logvar.data.uniform_(-1 / (in_features + out_features),
                                        1 / (in_features + out_features))
            self.bias_logvar.data.uniform_(-1 / out_features, 1 / out_features)

    def _zero_mean_forward(self, x):
        if self.deterministic:
            return self._det_forward(x)
        else:
            return self._mcvi_forward(x)

    def _get_var(self, param):
        if self.sigma_sq:
            return param * param
        else:
            return torch.exp(param)

    def compute_kl(self):
        return 0


class VarianceReluGaussian(ReluGaussian):
    def __init__(self, in_features, out_features,
                 certain=False, deterministic=True, sigma_sq=False):
        super().__init__(in_features, out_features, certain, deterministic)
        self.W.data.fill_(0)
        self.W.requires_grad = False
        self.sigma_sq = sigma_sq
        if sigma_sq:
            self.W_logvar.data.uniform_(-1 / (in_features + out_features),
                                        1 / (in_features + out_features))
            self.bias_logvar.data.uniform_(-1 / out_features, 1 / out_features)

    def _get_var(self, param):
        if self.sigma_sq:
            return param * param
        else:
            return torch.exp(param)

    def _zero_mean_forward(self, x):
        if self.deterministic:
            return self._det_forward(x)
        else:
            return self._mcvi_forward(x)

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

    def _get_var(self, param):
        return torch.exp(param)

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
        weights_kl = kl_gaussian(self.W, self._get_var(self.W_logvar),
                                 self.W_mean_prior, self.W_var_prior)
        bias_kl = kl_gaussian(self.bias, self._get_var(self.bias_logvar),
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

        W_var = self._get_var(self.W_logvar)
        bias_var = self._get_var(self.bias_logvar)

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

        W_var = self._get_var(self.W_logvar)
        bias_var = self._get_var(self.bias_logvar)

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

        W_var = self._get_var(self.W_logvar)
        bias_var = self._get_var(self.bias_logvar)

        z_mean = F.conv2d(x_mean, self.W, self.bias,
                          self.stride,
                          self.padding)
        z_var = F.conv2d(x_var, W_var, bias_var, self.stride,
                         self.padding)

        dst = Independent(Normal(z_mean, z_var), 1)
        sample = dst.rsample()
        return sample, None

    def _apply_activation(self, x):
        if self.activation == 'relu' and not self.certain:
            x_mean, x_var = x
            if x_var is None:
                x_var = x_mean * x_mean

            sqrt_x_var = torch.sqrt(x_var + EPS)
            mu = x_mean / sqrt_x_var
            z_mean = sqrt_x_var * softrelu(mu)
            z_var = x_var * (mu * standard_gaussian(mu) + (
                    1 + mu ** 2) * gaussian_cdf(mu))
            return z_mean, z_var
        else:
            return x

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', activation=' + str(self.activation) + ')'


class VarianceMeanFieldConv2d(MeanFieldConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 activation='relu', padding=0, certain=False,
                 deterministic=True, sigma_sq=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         activation, padding, certain, deterministic)
        self.W.data.fill_(0)
        self.W.requires_grad = False
        self.sigma_sq = sigma_sq
        if sigma_sq:
            self.W_logvar.data.uniform_(-1 / (in_channels + out_channels),
                                        1 / (in_channels + out_channels))
            self.bias_logvar.data.uniform_(-1 / out_channels, 1 / out_channels)

    def _get_var(self, param):
        if self.sigma_sq:
            return param * param
        else:
            return torch.exp(param)

    def compute_kl(self):
        return 0

    def _zero_mean_forward(self, x):
        if self.deterministic:
            return self._det_forward(x)
        else:
            return self._mcvi_forward(x)


class MeanFieldConv2dVDO(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, alpha_shape,
                 certain=False, activation='relu', deterministic=True, stride=1,
                 padding=0, dilation=1, prior='loguni', bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.dilation = dilation
        self.alpha_shape = alpha_shape
        self.groups = 1
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.op_bias = lambda input, kernel: F.conv2d(input, kernel,
                                                      self.bias.flatten(),
                                                      self.stride, self.padding,
                                                      self.dilation,
                                                      self.groups)
        self.op_nobias = lambda input, kernel: F.conv2d(input, kernel, None,
                                                        self.stride,
                                                        self.padding,
                                                        self.dilation,
                                                        self.groups)
        self.log_alpha = nn.Parameter(torch.Tensor(*alpha_shape))
        self.reset_parameters()

        self.certain = certain
        self.deterministic = deterministic
        self.mean_forward = False
        self.zero_mean = False
        self.permute_sigma = False

        self.prior = prior
        self.kl_fun = kl_loguni

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

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

    def _apply_activation(self, x):
        if self.activation == 'relu' and not self.certain:
            x_mean, x_var = x
            if x_var is None:
                x_var = x_mean * x_mean

            sqrt_x_var = torch.sqrt(x_var + EPS)
            mu = x_mean / sqrt_x_var
            z_mean = sqrt_x_var * softrelu(mu)
            z_var = x_var * (mu * standard_gaussian(mu) + (
                    1 + mu ** 2) * gaussian_cdf(mu))
            return z_mean, z_var
        else:
            return x

    def _zero_mean_forward(self, x):
        if self.certain or not self.deterministic:
            x_mean = x if not isinstance(x, tuple) else x[0]
            x_var = x_mean * x_mean
        else:
            x_mean = x[0]
            x_var = x[1]

        W_var = torch.exp(self.log_alpha) * self.weight * self.weight

        z_mean = F.conv2d(x_mean, torch.zeros_like(self.weight), self.bias,
                          self.stride,
                          self.padding)
        z_var = F.conv2d(x_var, W_var, bias=None, stride=self.stride,
                         padding=self.padding)

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

        z_mean = F.conv2d(x_mean, self.weight, self.bias,
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

        W_var = torch.exp(self.log_alpha) * self.weight * self.weight

        z_mean = F.conv2d(x_mean, self.weight, self.bias.flatten(),
                          self.stride,
                          self.padding)
        z_var = F.conv2d(x_var, W_var, bias=None, stride=self.stride,
                         padding=self.padding)
        return z_mean, z_var

    def _mcvi_forward(self, x):
        if isinstance(x, tuple):
            x_mean = x[0]
            x_var = x[1]
        else:
            x_mean = x
            x_var = x_mean * x_mean

        if self.zero_mean:
            lrt_mean = self.op_bias(x_mean, 0.0 * self.weight)
        else:
            lrt_mean = self.op_bias(x_mean, self.weight)

        sigma2 = torch.exp(self.log_alpha) * self.weight * self.weight
        if self.permute_sigma:
            sigma2 = sigma2.view(-1)[
                torch.randperm(self.weight.nelement()).cuda()].view(
                self.weight.shape)

        lrt_std = torch.sqrt(1e-16 + self.op_nobias(x_var, sigma2))
        if self.training:
            eps = lrt_std.data.new(lrt_std.size()).normal_()
        else:
            eps = 0.0
        return lrt_mean + lrt_std * eps, None

    def compute_kl(self):
        return self.weight.nelement() / self.log_alpha.nelement() * kl_loguni(
            self.log_alpha)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', alpha_shape=' + str(self.alpha_shape)
        s += ', prior=' + self.prior
        s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)


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

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)
