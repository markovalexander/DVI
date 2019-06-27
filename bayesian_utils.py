import torch
import numpy as np
import math

EPS = 1e-8


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
    arcsin_s = torch.abs(torch.asin(rho)) + EPS

    A = arcsin / (2 * math.pi)
    coef = rho_s / (2 * arcsin_s * rho_hat)
    coefs_prod = (rho * rho) / (arcsin_s * rho_hat * (1 + rho_hat))
    return A * torch.exp( -(mu1 * mu1 + mu2 * mu2) * coef + mu1 * mu2 * coefs_prod)


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

