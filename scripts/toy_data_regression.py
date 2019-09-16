import argparse
import math

import numpy as np
import torch
from torch import nn

from core.layers import LinearGaussian, ReluGaussian
from core.losses import RegressionLoss
from core.utils import draw_regression_result, get_predictions, \
    generate_regression_data

np.random.seed(42)

EPS = 1e-6

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--mcvi', action='store_true')
parser.add_argument('--hid_size', type=int, default=128)
parser.add_argument('--heteroskedastic', default=False, action='store_true')
parser.add_argument('--data_size', type=int, default=500)
parser.add_argument('--homo_var', type=float, default=0.35)
parser.add_argument('--homo_log_var_scale', type=float,
                    default=2 * math.log(0.2))
parser.add_argument('--method', type=str, default='bayes')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--anneal_updates', type=int, default=1000)
parser.add_argument('--warmup_updates', type=int, default=14000)
parser.add_argument('--test_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--gamma', type=float, default=0.5,
                    help='lr decrease rate in MultiStepLR scheduler')
parser.add_argument('--epochs', type=int, default=23000)
parser.add_argument('--draw_every', type=int, default=1000)
parser.add_argument('--milestones', nargs='+', type=int,
                    default=[3000, 5000, 9000, 13000])


def base_model(x):
    return -(x + 0.5) * np.sin(3 * np.pi * x)


def noise_model(x, args):
    if args.heteroskedastic:
        return 1 * (x + 0.5) ** 2
    else:
        return args.homo_var


def sample_data(x, args):
    return base_model(x) + np.random.normal(0, noise_model(x, args),
                                            size=x.size).reshape(x.shape)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        hid_size = args.hid_size
        self.linear = LinearGaussian(1, hid_size, certain=True)
        self.relu1 = ReluGaussian(hid_size, hid_size)
        if args.heteroskedastic:
            self.out = ReluGaussian(hid_size, 2)
        else:
            self.out = ReluGaussian(hid_size, 1)

        if args.mcvi:
            self.mcvi()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu1(x)
        return self.out(x)

    def mcvi(self):
        self.linear.mcvi()
        self.relu1.mcvi()
        self.out.mcvi()

    def determenistic(self):
        self.linear.determenistic()
        self.relu1.mcvi()
        self.out.mcvi()


if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    print(args)

    model = Model(args).to(args.device)
    loss = RegressionLoss(model, args)
    x_train, y_train, x_test, y_test, toy_data = generate_regression_data(args,
                                                                          sample_data,
                                                                          base_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     args.milestones,
                                                     gamma=args.gamma)

    if args.mcvi:
        mode = 'mcvi'
    else:
        mode = 'det'
    step = 0

    for epoch in range(args.epochs):
        step += 1
        optimizer.zero_grad()

        pred = model(x_train)
        neg_elbo, ll, kl = loss(pred, y_train, step)

        neg_elbo.backward()

        nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)
        scheduler.step()
        optimizer.step()

        if epoch % args.draw_every == 0:
            print("epoch : {}".format(epoch))
            print("ELBO : {:.4f}\t Likelihood: {:.4f}\t KL: {:.4f}".format(
                -neg_elbo.item(), ll.item(), kl.item()))

        if epoch % args.draw_every == 0:
            with torch.no_grad():
                predictions = get_predictions(x_train, model, args, args.mcvi)
                draw_regression_result(toy_data,
                                       {'mean': base_model,
                                        'std': lambda x: noise_model(x,
                                                                     args)},
                                       predictions=predictions,
                                       name='pics/{}/after_{}.png'.format(
                                           mode, epoch))

    with torch.no_grad():
        predictions = get_predictions(x_train, model, args, args.mcvi)
        draw_regression_result(toy_data,
                               {'mean': base_model,
                                'std': lambda x: noise_model(x, args)},
                               predictions=predictions,
                               name='pics/{}/last.png'.format(
                                   mode))

    if args.mcvi:
        model.determenistic()
        with torch.no_grad():
            predictions = get_predictions(x_train, model, args, False)
            draw_regression_result(toy_data,
                                   {'mean': base_model,
                                    'std': lambda x: noise_model(x, args)},
                                   predictions=predictions,
                                   name='pics/{}/swapped.png'.format(
                                       mode))
    else:
        model.mcvi()
        with torch.no_grad():
            predictions = get_predictions(x_train, model, args, True)
            draw_regression_result(toy_data,
                                   {'mean': base_model,
                                    'std': lambda x: noise_model(x, args)},
                                   predictions=predictions,
                                   name='pics/{}/swapped.png'.format(
                                       mode))
