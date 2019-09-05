import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import clip

from .bayesian_utils import logsoftmax_mean
from .utils import classification_posterior

EPS = 1e-8


class RegressionLoss(nn.Module):
    def __init__(self, net, args):
        """
        Compute ELBO for regression task

        :param net: neural network
        :param method:
        :param use_heteroskedastic:
        :param homo_log_var_scale:
        """
        super().__init__()

        self.net = net
        self.method = args.method
        self.use_het = args.heteroskedastic
        self.det = not args.mcvi
        self.homo_log_var_scale = torch.FloatTensor(
            [args.homo_log_var_scale]).to(
            device=args.device)
        if not self.use_het and self.homo_log_var_scale is None:
            raise ValueError(
                "homo_log_var_scale must be set in homoskedastic mode")
        self.warmup = args.warmup_updates
        self.anneal = args.anneal_updates
        self.batch_size = args.batch_size

    def gaussian_likelihood_core(self, target, mean, log_var, smm, sml, sll):
        const = math.log(2 * math.pi)
        exp = torch.exp(-log_var + 0.5 * (sll + EPS))
        return -0.5 * (
                const + log_var + exp * (smm + (mean - sml - target) ** 2))

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
        if self.det:
            mean = pred_mean[:, 0].view(-1)
        else:
            mean = pred_mean.view(-1)

        sll = sml = 0
        if self.method.lower() == 'bayes' and self.det:
            smm = pred_var[:, 0, 0].view(-1)
        else:
            smm = 0
        return self.gaussian_likelihood_core(target, mean, log_var, smm, sml,
                                             sll)

    def forward(self, pred, target, step):
        pred_mean = pred[0]
        pred_var = pred[1]

        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'compute_kl'):
                kl = kl + module.compute_kl()
        if hasattr(self.net, 'compute_kl'):
            kl = kl + self.net.compute_kl()

        gaussian_likelihood = self.heteroskedastic_gaussian_loglikelihood if self.use_het \
            else self.homoskedastic_gaussian_loglikelihood

        log_likelihood = gaussian_likelihood(pred_mean, pred_var, target)
        batched_likelihood = torch.mean(log_likelihood)

        lmbda = clip((step - self.warmup) / self.anneal, 0, 1)

        loss = lmbda * kl / self.batch_size - batched_likelihood
        return loss, batched_likelihood, kl / self.batch_size


class ClassificationLoss(nn.Module):
    def __init__(self, net, args):
        super().__init__()

        self.net = net
        self.warmup = args.warmup_updates
        self.anneal = args.anneal_updates
        self.data_size = args.data_size
        self.mcvi = args.mcvi
        self.use_samples = args.use_samples
        self.n_samples = args.mc_samples
        self._step = 0
        self.log_mean = vars(args).get('change_criterion', False)

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, logits, target):
        """
        Compute <log p(y | D)> - kl

        :param logits: shape [batch_size, n_classes]
        :param target: shape [batch_size, n_classes] -- one-hot target
        :param step:
        :return:
            total loss
            batch_logprob term
            total kl term
        """
        if not (self.mcvi or self.log_mean):
            logsoftmax = logsoftmax_mean(logits)
        elif self.mcvi:
            logsoftmax = F.log_softmax(logits[0], dim=1)
        elif self.log_mean:
            posterior = classification_posterior(logits)
            posterior = torch.clamp(posterior, 0, 1)
            posterior = posterior / torch.sum(posterior, dim=-1, keepdim=True)
            logsoftmax = torch.log(posterior + EPS)

        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'compute_kl'):
                kl = kl + module.compute_kl()

        logprob = torch.sum(target.type(logsoftmax.type()) * logsoftmax, dim=1)
        batch_logprob = torch.mean(logprob)

        lmbda = clip((self._step - self.warmup) / self.anneal, 0, 1)
        L = lmbda * kl / self.data_size - batch_logprob
        return L, batch_logprob, kl / self.data_size, logsoftmax

    def step(self):
        self._step += 1
