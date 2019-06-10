import math
import torch
import torch.nn as nn

EPS = 1e-8


class RegressionLoss(nn.Module):
    def __init__(self, net, method='bayes', use_heteroskedastic=False,
                 homo_log_var_scale=None):
        """
        Compute ELBO for regression task

        :param net: neural network
        :param method:
        :param use_heteroskedastic:
        :param homo_log_var_scale:
        """
        super().__init__()

        self.net = net
        self.method = method
        self.use_het = use_heteroskedastic
        if not self.use_het and homo_log_var_scale is None:
            raise ValueError(
                "homo_log_var_scale must be set in homoskedastic mode")

        if self.use_het:
            raise NotImplementedError("heterostadic is not supported yet")

        self.homo_log_var_scale = torch.FloatTensor([homo_log_var_scale])

    def gaussian_likelihood_core(self, target, mean, log_var, smm, sml, sll):
        const = math.log(2 * math.pi)
        exp = torch.exp(-log_var + 0.5 * sll)
        return const + log_var + exp * (smm + (mean - sml - target) ** 2)

    def heteroskedastic_gaussian_loglikelihood(self, pred_mean, pred_var,
                                               target):
        log_var = pred_mean[:, 1].view(-1)
        mean = pred_mean[:, 0].view(-1)

        if self.method.lower() == 'bayes':
            sll = pred_var[:, 1, 1].view(-1)
            smm = pred_var[:, 0, 0].view(-1)
            sml = pred_var[:, 0, 1].view(-1)
        else:
            sll = smm = sml = 0
        return self.gaussian_likelihood_core(target, mean, log_var, smm, sml,
                                             sll)

    def homoskedastic_gaussian_loglikelihood(self, pred_mean, pred_var, target):
        log_var = self.homo_log_var_scale
        mean = pred_mean[:, 0].view(-1)
        sll = sml = 0
        if self.method.lower() == 'bayes':
            smm = pred_var[:, 0, 0].view(-1)
        else:
            smm = 0
        return self.gaussian_likelihood_core(target, mean, log_var, smm, sml,
                                             sll)

    def forward(self, pred, target):
        pred_mean = pred[0]
        pred_var = pred[1]

        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'compute_kl'):
                kl = kl + module.compute_kl()

        gaussian_likelihood = self.heteroskedastic_gaussian_loglikelihood if self.use_het \
            else self.homoskedastic_gaussian_loglikelihood

        log_likelihood = gaussian_likelihood(pred_mean, pred_var, target)
        batched_likelihood = torch.mean(log_likelihood)

        loss = kl - batched_likelihood
        return loss, batched_likelihood.detach(), kl
