import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

EPS = 1e-6


def matrix_diag_part(tensor):
    return torch.diagonal(tensor, dim1=-1, dim2=-2)


def standard_gaussian(x):
    return torch.exp(-1 / 2 * x * x) / np.sqrt(2 * math.pi)


def gaussian_cdf(x):
    const = 1 / np.sqrt(2)
    return 0.5 * (1 + torch.erf(x * const))


def softrelu(x):
    return standard_gaussian(x) + x * gaussian_cdf(x)


def heaviside_q(rho, mu1, mu2):
    """
    Compute exp ( -Q(rho, mu1, mu2) ) for Heaviside activation

    """
    rho_hat = torch.sqrt(1 - rho * rho)
    arcsin = torch.asin(rho)

    rho_s = torch.abs(rho) + EPS
    arcsin_s = torch.abs(torch.asin(rho)) + EPS / 2

    A = arcsin / (2 * math.pi)
    one_over_coef_sum = (2 * arcsin_s * rho_hat) / rho_s
    one_over_coefs_prod = (arcsin_s * rho_hat * (1 + rho_hat)) / (rho * rho)
    return A * torch.exp(-(
            mu1 * mu1 + mu2 * mu2) / one_over_coef_sum + mu1 * mu2 / one_over_coefs_prod)


def relu_q(rho, mu1, mu2):
    """
    Compute exp ( -Q(rho, mu1, mu2) ) for ReLU activation

    """
    rho_hat_plus_one = torch.sqrt(1 - rho * rho) + 1
    g_r = torch.asin(rho) - rho / rho_hat_plus_one  # why minus? why no brackets

    rho_s = torch.abs(rho) + EPS
    g_r_s = torch.abs(g_r) + EPS
    A = g_r / (2 * math.pi)

    coef_sum = rho_s / (2 * g_r_s * rho_hat_plus_one)
    coef_prod = (torch.asin(rho) - rho) / (rho_s * g_r_s)
    return A * torch.exp(
        - (mu1 * mu1 + mu2 * mu2) * coef_sum + coef_prod * mu1 * mu2)


def delta(rho, mu1, mu2):
    return gaussian_cdf(mu1) * gaussian_cdf(mu2) + relu_q(rho, mu1, mu2)


def compute_linear_var(x_mean, x_var, weights_mean, weights_var,
                       bias_mean=None, bias_var=None):
    x_var_diag = matrix_diag_part(x_var)
    xx_mean = x_var_diag + x_mean * x_mean

    term1_diag = torch.matmul(xx_mean, weights_var)

    flat_xCov = torch.reshape(x_var, (-1, weights_mean.size(0)))

    xCov_A = torch.matmul(flat_xCov, weights_mean)
    xCov_A = torch.reshape(xCov_A, (
        -1, weights_mean.size(0), weights_mean.size(1)))

    xCov_A = torch.transpose(xCov_A, 1, 2)
    xCov_A = torch.reshape(xCov_A, (-1, weights_mean.size(0)))

    A_xCov_A = torch.matmul(xCov_A, weights_mean)
    A_xCov_A = torch.reshape(A_xCov_A, (
        -1, weights_mean.size(1), weights_mean.size(1)))

    term2 = A_xCov_A
    term2_diag = matrix_diag_part(term2)

    _, n, _ = term2.size()
    idx = torch.arange(0, n)

    term3_diag = bias_var if bias_var is not None else 0
    result_diag = term1_diag + term2_diag + term3_diag

    result = term2
    result[:, idx, idx] = result_diag
    return result


def compute_heaviside_var(x_var, x_var_diag, mu):
    mu1 = torch.unsqueeze(mu, 2)
    mu2 = mu1.permute(0, 2, 1)

    s11s22 = torch.unsqueeze(x_var_diag, dim=2) * torch.unsqueeze(
        x_var_diag, dim=1)
    rho = x_var / torch.sqrt(s11s22)
    rho = torch.clamp(rho, -1 / (1 + 1e-6), 1 / (1 + 1e-6))
    return heaviside_q(rho, mu1, mu2)


def compute_relu_var(x_var, x_var_diag, mu):
    mu1 = torch.unsqueeze(mu, 2)
    mu2 = mu1.permute(0, 2, 1)

    s11s22 = torch.unsqueeze(x_var_diag, dim=2) * torch.unsqueeze(
        x_var_diag, dim=1)
    rho = x_var / (torch.sqrt(s11s22) + EPS)
    rho = torch.clamp(rho, -1 / (1 + EPS), 1 / (1 + EPS))
    return x_var * delta(rho, mu1, mu2)


def kl_gaussian(p_mean, p_var, q_mean, q_var):
    """
    Computes KL (p || q) from p to q, assuming that both p and q have diagonal
    gaussian distributions

    :param p_mean:
    :param p_var:
    :param prior:
    :return:
    """
    s_q_var = q_var + EPS
    entropy = 0.5 * (1 + math.log(2 * math.pi) + torch.log(p_var))
    cross_entropy = 0.5 * (math.log(2 * math.pi) + torch.log(s_q_var) + \
                           (p_var + (p_mean - q_mean) ** 2) / s_q_var)
    return torch.sum(cross_entropy - entropy)


def kl_loguni(log_alpha):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    C = -k1
    mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(
        torch.exp(-log_alpha)) + C
    kl = -torch.sum(mdkl)
    return kl


def logsumexp_mean(y, keepdim=True):
    """
    Compute <logsumexp(y)>
    :param y: tuple of (y_mean, y_var)
    y_mean dim [batch_size, hid_dim]
    y_var dim  [batch_size, hid_dim, hid_dim]
    :return:
    """
    y_mean = y[0]
    y_var = y[1]
    logsumexp = torch.logsumexp(y_mean, dim=-1, keepdim=keepdim)
    p = torch.exp(y_mean - logsumexp)

    pTdiagVar = torch.sum(p * matrix_diag_part(y_var), dim=-1, keepdim=keepdim)
    pTVarp = torch.squeeze(torch.matmul(torch.unsqueeze(p, 1),
                                        torch.matmul(y_var,
                                                     torch.unsqueeze(p, 2))),
                           dim=-1)

    return logsumexp + 0.5 * (pTdiagVar - pTVarp)


def logsoftmax_mean(y):
    """
    Compute <logsoftmax(y)>
    :param y:
    :param y: tuple of (y_mean, y_var)
    y_mean dim [batch_size, hid_dim]
    y_var dim  [batch_size, hid_dim, hid_dim]

    """
    return y[0] - logsumexp_mean(y)


def sample_activations(x, n_samples):
    x_mean, x_var = x[0], x[1]
    sampler = MultivariateNormal(loc=x_mean, covariance_matrix=x_var)
    samples = sampler.rsample([n_samples])
    return samples


def sample_logsoftmax(x, n_samples):
    activations = sample_activations(x, n_samples)
    logsoftmax = F.log_softmax(activations, dim=1)
    return torch.mean(logsoftmax, dim=0)


def sample_softmax(x, n_samples):
    activations = sample_activations(x, n_samples)
    softmax = F.softmax(activations, dim=1)
    return torch.mean(softmax, dim=0)


def classification_posterior(activations):
    mean, var = activations
    p = F.softmax(mean, dim=1)
    diagVar = matrix_diag_part(var)
    pTdiagVar = torch.sum(p * diagVar, dim=-1, keepdim=True)
    pTVarp = torch.squeeze(torch.matmul(torch.unsqueeze(p, 1),
                                        torch.matmul(var,
                                                     torch.unsqueeze(p, 2))),
                           dim=-1)
    Varp = torch.squeeze(torch.matmul(var, torch.unsqueeze(p, 2)), dim=-1)

    return p * (1 + pTVarp - Varp + 0.5 * diagVar - 0.5 * pTdiagVar)
