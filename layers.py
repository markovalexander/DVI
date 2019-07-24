import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from bayesian_utils import KL_GG, softrelu, delta, heaviside_q, gaussian_cdf, \
    matrix_diag_part

EPS = 1e-6


class LinearGaussian(nn.Module):
    def __init__(self, in_features, out_features, certain=False,
                 prior="DiagonalGaussian", device='cpu'):
        """
        Applies linear transformation y = xA^T + b

        A and b are Gaussian random variables

        :param in_features: input dimension
        :param out_features: output dimension
        :param certain:  if false, than x is equal to its mean and has no variance
        :param prior:  prior type
        """

        super().__init__()

        self.A_mean = nn.Parameter(torch.Tensor(in_features, out_features))
        self.b_mean = nn.Parameter(torch.Tensor(out_features))
        self.certain = certain

        self.A_logvar = nn.Parameter(torch.Tensor(in_features, out_features))
        self.b_logvar = nn.Parameter(torch.Tensor(out_features))

        self.prior = prior
        self.initialize_weights()
        self.construct_priors(self.prior, device)
        self.use_dvi = True

    def initialize_weights(self):
        nn.init.xavier_normal_(self.A_mean)
        nn.init.normal_(self.b_mean)

        nn.init.xavier_normal_(self.A_logvar)
        nn.init.normal_(self.b_logvar)

    def construct_priors(self, prior, device):
        if prior == "DiagonalGaussian":
            s1 = s2 = 0.1

            self._prior_A = {
                'mean': torch.zeros_like(self.A_mean, requires_grad=False).to(
                    device),
                'var': torch.ones_like(self.A_logvar, requires_grad=False).to(
                    device) * s2}
            self._prior_b = {
                'mean': torch.zeros_like(self.b_mean, requires_grad=False).to(
                    device),
                'var': torch.ones_like(self.b_logvar, requires_grad=False).to(
                    device) * s1}
        else:
            raise NotImplementedError("{} prior is not supported".format(prior))

    def compute_kl(self):
        if self.prior == 'DiagonalGaussian':
            kl_A = KL_GG(self.A_mean, torch.exp(self.A_logvar),
                         self._prior_A['mean'].to(self.A_mean.device),
                         self._prior_A['var'].to(self.A_mean.device))
            kl_b = KL_GG(self.b_mean, torch.exp(self.b_logvar),
                         self._prior_b['mean'].to(self.A_mean.device),
                         self._prior_b['var'].to(self.A_mean.device))
        return kl_A + kl_b

    def determenistic(self, mode=True):
        self.use_dvi = True

    def mcvi(self, mode=True):
        self.use_dvi = not mode

    def get_mode(self):
        if self.use_dvi:
            print('In determenistic mode')
        else:
            print('In MCVI mode')

    def forward(self, x):
        """
        Compute expectation and variance after linear transform
        y = xA^T + b

        :param x: input, size [batch, in_features]
        :return: tuple (y_mean, y_var) for determenistic mode:,  shapes:
                 y_mean: [batch, out_features]
                 y_var:  [batch, out_features, out_features]

                 tuple (sample, None) for MCVI mode,
                 sample : [batch, out_features] - local reparametrization of output
        """
        A_var = torch.exp(self.A_logvar)
        b_var = torch.exp(self.b_logvar)
        if self.use_dvi:
            return self._det_forward(x, A_var, b_var)
        else:
            return self._mcvi_forward(x, A_var, b_var)

    def _mcvi_forward(self, x, A_var, b_var):
        if self.certain:
            x_mean = x
            x_var = None
        else:
            x_mean = x[0]
            x_var = x[1]

        y_mean = F.linear(x_mean, self.A_mean.t()) + self.b_mean

        if self.certain or not self.use_dvi:
            xx = x_mean * x_mean
            y_var = torch.diag_embed(F.linear(xx, A_var.t()) + b_var)
        else:
            y_var = self.compute_var(x_mean, x_var)

        dst = MultivariateNormal(loc=y_mean, covariance_matrix=y_var)
        sample = dst.rsample()
        return sample, None

    def _det_forward(self, x, A_var, b_var):
        """
        Compute expectation and variance after linear transform
        y = xA^T + b

        :param x: input, size [batch, in_features]
        :return: tuple (y_mean, y_var),  shapes:
                 y_mean: [batch, out_features]
                 y_var:  [batch, out_features, out_features]
        """

        if self.certain:
            x_mean = x
            x_var = None
        else:
            x_mean = x[0]
            x_var = x[1]

        y_mean = F.linear(x_mean, self.A_mean.t()) + self.b_mean

        if self.certain:
            xx = x_mean * x_mean
            y_var = torch.diag_embed(F.linear(xx, A_var.t()) + b_var)
        else:
            y_var = self.compute_var(x_mean, x_var)

        return y_mean, y_var

    def compute_var(self, x_mean, x_var):
        A_var = torch.exp(self.A_logvar)

        x_var_diag = matrix_diag_part(x_var)
        xx_mean = x_var_diag + x_mean * x_mean

        term1_diag = torch.matmul(xx_mean, A_var)

        flat_xCov = torch.reshape(x_var, (-1, self.A_mean.size(0)))  # [b*x, x]
        xCov_A = torch.matmul(flat_xCov, self.A_mean)  # [b * x, y]
        xCov_A = torch.reshape(xCov_A, (
            -1, self.A_mean.size(0), self.A_mean.size(1)))  # [b, x, y]
        xCov_A = torch.transpose(xCov_A, 1, 2)  # [b, y, x]
        xCov_A = torch.reshape(xCov_A, (-1, self.A_mean.size(0)))  # [b*y, x]

        A_xCov_A = torch.matmul(xCov_A, self.A_mean)  # [b*y, y]
        A_xCov_A = torch.reshape(A_xCov_A, (
            -1, self.A_mean.size(1), self.A_mean.size(1)))  # [b, y, y]

        term2 = A_xCov_A
        term2_diag = matrix_diag_part(term2)

        _, n, _ = term2.size()
        idx = torch.arange(0, n)

        term3_diag = torch.exp(self.b_logvar)
        result_diag = term1_diag + term2_diag + term3_diag

        result = term2
        result[:, idx, idx] = result_diag
        return result


class ReluGaussian(nn.Module):
    def __init__(self, in_features, out_features, certain=False,
                 prior="DiagonalGaussian", device='cpu'):
        """
        Computes y = relu(x) * A.T + b

        A and b are Gaussian random variables

        :param in_features: input dimension
        :param out_features: output dimension
        :param certain:  if false, than x is equal to its mean and has no variance
        :param prior:  prior type
        """

        super().__init__()
        self.linear = LinearGaussian(in_features, out_features, certain, prior,
                                     device=device)
        self.certain = certain
        self.use_dvi = True

    def compute_kl(self):
        return self.linear.compute_kl()

    def determenistic(self, mode=True):
        self.linear.use_dvi = mode
        self.use_dvi = mode

    def mcvi(self, mode=True):
        self.linear.use_dvi = not mode
        self.use_dvi = not mode

    def forward(self, x):
        x_mean = x[0]
        x_var = x[1]

        if not self.use_dvi:
            z_mean = F.relu(x_mean)
            z_var = None
        else:
            x_var_diag = matrix_diag_part(x_var)
            sqrt_x_var_diag = torch.sqrt(x_var_diag + EPS)
            mu = x_mean / (sqrt_x_var_diag + EPS)

            z_mean = sqrt_x_var_diag * softrelu(mu)
            z_var = self.compute_var(x_var, x_var_diag, mu)

        return self.linear((z_mean, z_var))

    def compute_var(self, x_var, x_var_diag, mu):
        mu1 = torch.unsqueeze(mu, 2)
        mu2 = mu1.permute(0, 2, 1)

        s11s22 = torch.unsqueeze(x_var_diag, dim=2) * torch.unsqueeze(
            x_var_diag, dim=1)
        rho = x_var / (torch.sqrt(s11s22) + EPS)
        rho = torch.clamp(rho, -1 / (1 + EPS), 1 / (1 + EPS))
        return x_var * delta(rho, mu1, mu2)

    def get_mode(self):
        if self.linear.use_dvi:
            print('Using determenistic mode')
        else:
            print('Using MCVI')


class HeavisideGaussian(nn.Module):
    def __init__(self, in_features, out_features, certain=False,
                 prior="DiagonalGaussian"):
        """
        Computes y = heaviside(x) * A.T + b

        A and b are Gaussian random variables

        :param in_features: input dimension
        :param out_features: output dimension
        :param certain:  if false, than x is equal to its mean and has no variance
        :param prior:  prior type
        """

        super().__init__()
        self.linear = LinearGaussian(in_features, out_features, certain, prior)
        self.certain = certain
        self.use_dvi = True

    def compute_kl(self):
        return self.linear.compute_kl()

    def determenistic(self, mode=True):
        self.linear.use_dvi = mode
        self.use_dvi = mode

    def mcvi(self, mode=True):
        self.linear.use_dvi = not mode
        self.use_dvi = not mode

    def forward(self, x):
        if not self.use_dvi:
            raise NotImplementedError(
                "Heaviside activation function is not supported in MCVI mode")
        #
        # if not self.use_dvi:
        #     x_mean = x[0]
        #     x_var = x_mean * x_mean
        #
        #     z_mean = x_mean
        #     z_mean[x_mean > 0] = 1
        #     z_mean[]
        x_mean = x[0]
        x_var = x[1]

        x_var_diag = matrix_diag_part(x_var)

        sqrt_x_var_diag = torch.sqrt(x_var_diag)
        mu = x_mean / (sqrt_x_var_diag + EPS)

        z_mean = gaussian_cdf(mu)
        z_var = self.compute_var(x_var, x_var_diag, mu)

        return self.linear((z_mean, z_var))

    def compute_var(self, x_var, x_var_diag, mu):
        mu1 = torch.unsqueeze(mu, 2)
        mu2 = mu1.permute(0, 2, 1)

        s11s22 = torch.unsqueeze(x_var_diag, dim=2) * torch.unsqueeze(
            x_var_diag, dim=1)
        rho = x_var / torch.sqrt(s11s22)
        rho = torch.clamp(rho, -1 / (1 + 1e-6), 1 / (1 + 1e-6))
        return heaviside_q(rho, mu1, mu2)

    def get_mode(self):
        if self.linear.use_dvi:
            print('Using determenistic mode')
        else:
            print('Using MCVI')


class MeanFieldConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0,
                 prior='DiagonalGaussian', certain=False):
        super().__init__()

        self.use_det = True
        self.certain = certain
        self.stride = stride
        self.padding = padding

        self.weights_mean = nn.Parameter(
            torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))
        self.weights_log_var = nn.Parameter(
            torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))

        self.bias_mean = nn.Parameter(torch.Tensor(out_channels))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_channels))

        self.prior = prior
        self.initialize_weights()
        self.construct_priors()

    def construct_priors(self):
        if self.prior == "DiagonalGaussian":

            s1 = s2 = 0.1
            self._weight_prior = {
                'mean': nn.Parameter(torch.zeros_like(self.weights_mean), requires_grad=False),
                'var': nn.Parameter(torch.ones_like(self.weights_log_var), requires_grad=False) * s1
            }
            self._bias_prior = {
                'mean': nn.Parameter(torch.zeros_like(self.bias_mean, requires_grad=False)),
                'var': nn.Parameter(torch.ones_like(self.bias_log_var), requires_grad=False) * s2
            }
        else:
            raise NotImplementedError("{} prior is not supported".format(self.prior))

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.weights_mean)
        nn.init.kaiming_normal_(self.b_mean)

        nn.init.kaiming_normal_(self.A_logvar)
        nn.init.kaiming_normal_(self.b_logvar)

    def compute_kl(self):
        weights_kl = KL_GG(self.weights_mean, torch.exp(self.weights_log_var),
                           self._weight_prior['mean'], self._weight_prior['var'])
        bias_kl = KL_GG(self.bias_mean, torch.exp(self.bias_log_var),
                        self._bias_prior['mean'], self._bias_prior['var'])
        return weights_kl + bias_kl

    def get_mode(self):
        if self.use_det:
            return "Determenistic"
        else:
            return "MonteCarlo"

    def determenistic(self, mode=True):
        self.use_det = mode

    def mcvi(self, mode=True):
        self.use_det = not mode

    def forward(self, x):
        if self.use_det:
            return self.__det_forward(x)
        else:
            return self.__mcvi_forward(x)

    def __det_forward(self, x):

        if self.certain:
            x_mean = x[0]
            x_var = x_mean * x_mean
        else:
            x_mean = x[0]
            x_var = x[1]

        weights_var = torch.exp(self.weights_log_var)
        bias_var = torch.exp(self.bias_log_var)

        z_mean = F.conv2d(x_mean, self.weights_mean, self.bias_mean, self.stride,
                          self.padding)
        z_var = F.conv2d(x_var, weights_var, bias_var, self.stride, self.padding)
        return z_mean, z_var


    def __mcvi_forward(self, x):
        raise NotImplementedError()
