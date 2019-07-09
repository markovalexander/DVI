import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from layers import LinearGaussian, ReluGaussian
from losses import RegressionLoss

import argparse

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
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--anneal_updates', type=int, default=1000)
parser.add_argument('--warmup_updates', type=int, default=14000)
parser.add_argument('--test_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--gamma', type=float, default=0.5,
                    help='lr decrease rate in MultiStepLR scheduler')
parser.add_argument('--epochs', type=int, default=23000)


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


def generate_data(args):
    data_size = {'train': args.data_size, 'test': args.test_size}
    toy_data = []
    for section in ['train', 'test']:
        x = (np.random.rand(data_size['train'], 1) - 0.5)
        toy_data.append([x, sample_data(x, args).reshape(-1)])
    x = np.arange(-1, 1, 1 / 100)
    toy_data.append([[[_] for _ in x], base_model(x)])
    x_train = toy_data[0][0]
    y_train = toy_data[0][1]

    x_test = toy_data[1][0]
    y_test = toy_data[1][1]

    x_train = torch.FloatTensor(x_train).to(device=args.device)
    y_train = torch.FloatTensor(y_train).to(device=args.device)

    x_test = torch.FloatTensor(x_test).to(device=args.device)
    y_test = torch.FloatTensor(y_test).to(device=args.device)

    return x_train, y_train, x_test, y_test, toy_data


def get_predictions(data, model):
    output = model(data)

    output_cov = output[1]
    output_mean = output[0]

    n = output_mean.size(0)
    m = output_mean.size(1)

    out_cov = torch.reshape(output_cov, (n, m, m))
    out_mean = output_mean

    x = data.cpu().detach().numpy().squeeze()
    y = {'mean': out_mean.cpu().detach().numpy(),
         'cov': out_cov.cpu().detach().numpy()}
    return (x, y)


def toy_results_plot(data, data_generator, predictions=None, name=None):
    x = predictions[0]
    train_x = np.arange(np.min(x.reshape(-1)),
                        np.max(x.reshape(-1)), 1 / 100)

    # plot the training data distribution
    plt.figure(figsize=(14, 10))

    plt.plot(train_x, data_generator['mean'](train_x), 'red', label='data mean')
    plt.fill_between(train_x,
                     data_generator['mean'](train_x) - data_generator['std'](
                         train_x),
                     data_generator['mean'](train_x) + data_generator['std'](
                         train_x),
                     color='orange', alpha=1, label='data 1-std')
    plt.plot(data[0][0], data[0][1], 'r.', alpha=0.2, label='train sampl')

    # plot the model distribution
    if predictions is not None:
        x = predictions[0]
        y_mean = predictions[1]['mean'][:, 0]
        ell_mean = 2 * math.log(0.2)
        y_var = predictions[1]['cov'][:, 0, 0]
        ell_var = 0

        heteroskedastic_part = np.exp(0.5 * ell_mean)
        full_std = np.sqrt(y_var + np.exp(ell_mean + 0.5 * ell_var))

        plt.scatter(x, y_mean, label='model mean')
        plt.scatter(x, y_mean - heteroskedastic_part, color='g', alpha=0.2)
        plt.scatter(x, y_mean + heteroskedastic_part, color='g', alpha=0.2,
                    label='$\ell$ contrib')
        # plt.fill_between(x,
        #                  y_mean - heteroskedastic_part,
        #                  y_mean + heteroskedastic_part,
        #                  color='g', alpha=0.2, label='$\ell$ contrib')

        plt.scatter(x, y_mean - full_std, color='b', alpha=0.2,
                    label='model 1-std')
        plt.scatter(x, y_mean + full_std, color='b', alpha=0.2)
        # plt.fill_between(x,
        #                  y_mean - full_std,
        #                  y_mean + full_std,
        #                  color='b', alpha=0.2, label='model 1-std')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([-3, 2])
    plt.legend()
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model = Model(args).to(args.device)
    loss = RegressionLoss(model, args)
    x_train, y_train, x_test, y_test, toy_data = generate_data(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [3000, 5000, 9000, 13000],
                                                     gamma=args.gamma)

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

        if epoch % 1000 == 0:
            print("epoch : {}".format(epoch))
            print("ELBO : {:.4f}\t Likelihood: {:.4f}\t KL: {:.4f}".format(
                -neg_elbo.item(), ll.item(), kl.item()))

        if epoch % 1000 == 0:
            with torch.no_grad():
                predictions = get_predictions(x_train, model)
                toy_results_plot(toy_data,
                                 {'mean': base_model,
                                  'std': lambda x: noise_model(x, args)},
                                 predictions=predictions,
                                 name='pics/toy_data/after_{}.png'.format(
                                     epoch))
