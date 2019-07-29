import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

EPS = 1e-6


def matrix_diag_part(tensor):
    return torch.stack(tuple(t.diag() for t in torch.unbind(tensor, 0)))


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
    g_r = torch.asin(rho) - rho / rho_hat_plus_one  # why minus?

    rho_s = torch.abs(rho) + EPS
    g_r_s = torch.abs(g_r) + EPS
    A = g_r / (2 * math.pi)

    coef_sum = rho_s / (2 * g_r_s * rho_hat_plus_one)
    coef_prod = (torch.asin(rho) - rho) / (rho_s * g_r_s)
    return A * torch.exp(
        - (mu1 * mu1 + mu2 * mu2) * coef_sum + coef_prod * mu1 * mu2)


def delta(rho, mu1, mu2):
    return gaussian_cdf(mu1) * gaussian_cdf(mu2) + relu_q(rho, mu1, mu2)


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


def sample_logsoftmax(logits, n_samples):
    activations = sample_activations(logits, n_samples)
    logsoftmax = F.log_softmax(activations, dim=1)
    return torch.mean(logsoftmax, dim=0)


def sample_softmax(logits, n_samples):
    activations = sample_activations(logits, n_samples)
    softmax = F.softmax(activations, dim=1)
    return torch.mean(softmax, dim=0)


def classification_posterior(mean, var):
    p = F.softmax(mean, dim=1)

    diagVar = matrix_diag_part(var)
    pTdiagVar = torch.sum(p * diagVar, dim=-1, keepdim=True)
    pTVarp = torch.squeeze(torch.matmul(torch.unsqueeze(p, 1),
                                        torch.matmul(var,
                                                     torch.unsqueeze(p, 2))),
                           dim=-1)
    Varp = torch.squeeze(torch.matmul(var, torch.unsqueeze(p, 2)), dim=-1)

    return p * (1 + pTVarp - Varp + 0.5 * diagVar - 0.5 * pTdiagVar)
