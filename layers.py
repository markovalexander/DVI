import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def matrix_diag_part(tensor):
    return torch.stack(tuple(t.diag() for t in torch.unbind(tensor, 0)))


def KL_GG(p_mean, p_var, q_mean, q_var):
    """
    Computes KL (p || q) from p to q, assuming that both p and q have normal
    distribution

    :param p_mean:
    :param p_var:
    :param q_mean:
    :param q_var:
    :return:
    """
    s_q_var = q_var + EPS
    entropy = 0.5 * (1 + math.log(2 * math.pi) + torch.log(p_var))
    cross_entropy = 0.5 * (math.log(2 * math.pi) + torch.log(s_q_var) + \
                           (p_var + (p_mean - q_mean) ** 2) / s_q_var)
    return torch.sum(cross_entropy - entropy)


class LinearGaussian(nn.Module):
    def __init__(self, in_features, out_features, certain=False, prior=None):
        """
        Applies linear transformation y = xA^T + b

        A and b are Gaussian random variables

        :param in_features: input dimension
        :param out_features: output dimension
        :param certain:  if false, than x is equal to its mean and has no variance
        :param prior:  prior type
        """

        super().__init__()

        self.A_mean = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.b_mean = nn.Parameter(torch.Tensor(out_dim))
        self.certain = certain

        self.A_var = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.b_var = nn.Parameter(torch.Tensor(out_dim))

        self.prior = prior
        self.initialize_weights()
        self.construct_priors(self.prior)

    def initialize_weights(self):
        nn.init.zeros_(self.A_mean)
        nn.init.zeros_(self.b_mean)

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

    def forward(self, x):
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

        term3_diag = self.b_var
        result_diag = term1_diag + term2_diag + term3_diag
        return torch.diag_embed(result_diag)
