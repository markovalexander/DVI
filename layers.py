import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal

from bayesian_utils import KL_GG

EPS = 1e-8


def matrix_diag_part(tensor):
    return torch.stack(tuple(t.diag() for t in torch.unbind(tensor, 0)))


class LinearGaussian(nn.Module):
    def __init__(self, in_features, out_features, certain=False,
                 prior="DiagonalGaussian"):
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

        self.A_var = nn.Parameter(torch.Tensor(in_features, out_features))
        self.b_var = nn.Parameter(torch.Tensor(out_features))

        self.prior = prior
        self.initialize_weights()
        self.construct_priors(self.prior)
        self.use_dvi = True

    def initialize_weights(self):
        nn.init.xavier_normal_(self.A_mean)
        nn.init.normal_(self.b_mean)

        shape = self.b_var.size(0)
        s = shape * shape

        nn.init.uniform_(self.A_var, a=0, b=s)
        nn.init.uniform_(self.b_var, a=0, b=s)

    def construct_priors(self, prior):
        if prior == "DiagonalGaussian":
            s1 = 1
            s2 = 1

            self._prior_A = {
                'mean': torch.zeros_like(self.A_mean, requires_grad=False),
                'var': torch.ones_like(self.A_var, requires_grad=False) * s2}
            self._prior_b = {
                'mean': torch.zeros_like(self.b_mean, requires_grad=False),
                'var': torch.ones_like(self.b_var, requires_grad=False) * s1}
        else:
            raise NotImplementedError("{} prior is not supported".format(prior))

    def compute_kl(self):
        if self.prior == 'DiagonalGaussian':
            kl_A = KL_GG(self.A_mean, self.A_var, self._prior_A['mean'],
                         self._prior_A['var'])
            kl_b = KL_GG(self.b_mean, self.b_var, self._prior_b['mean'],
                         self._prior_b['var'])
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
        if self.use_dvi:
            return self._det_forward(x)
        else:
            return self._mcvi_forward(x)

    def _mcvi_forward(self, x):
        if self.certain or not self.use_dvi:
            x_mean = x
            x_var = None
        else:
            x_mean = x[0]
            x_var = x[1]

        y_mean = F.linear(x_mean, self.A_mean.t()) + self.b_mean

        if self.certain or not self.use_dvi:
            xx = x_mean * x_mean
            y_var = torch.diag_embed(F.linear(xx, self.A_var.t()) + self.b_var)
        else:
            y_var = self.compute_var(x_mean, x_var)

        dst = MultivariateNormal(loc=y_mean, covariance_matrix=y_var)
        sample = dst.rsample()
        return sample, None

    def _det_forward(self, x):
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
            y_var = torch.diag_embed(F.linear(xx, self.A_var.t()) + self.b_var)
        else:
            y_var = self.compute_var(x_mean, x_var)

        return y_mean, y_var


    def compute_var(self, x_mean, x_var):
        x_var_diag = matrix_diag_part(x_var)
        xx_mean = x_var_diag + x_mean * x_mean

        term1_diag = torch.matmul(xx_mean, self.A_var)

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

        term3_diag = self.b_var
        result_diag = term1_diag + term2_diag + term3_diag

        result = term2
        result[:, idx, idx] = result_diag
        return result
