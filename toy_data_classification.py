import argparse

import numpy as np
import torch
from torch import nn

from layers import LinearGaussian, ReluGaussian
from losses import ClassificationLoss

from utils import generate_classification_data, draw_classification_results


np.random.seed(42)

EPS = 1e-6

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--mcvi', action='store_true')
parser.add_argument('--hid_size', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--data_size', type=int, default=500)
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
parser.add_argument('--dataset', default='classification')
parser.add_argument('--input_size', default=2, type=int)
parser.add_argument('--mc_samples', default=1, type=int)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        hid_size = args.hid_size
        self.linear = LinearGaussian(args.input_size, hid_size, certain=True)
        self.relu1 = ReluGaussian(hid_size, hid_size)
        self.out = ReluGaussian(hid_size, args.n_classes)

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
    args.device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    print(args)

    x_train, y_train, y_onehot_train, x_test, y_test, y_onehot_test = generate_classification_data(args)
    draw_classification_results(x_test, y_test, 'test.png', args)
    draw_classification_results(x_train, y_train, 'train.png', args)

    model = Model(args).to(args.device)
    loss = ClassificationLoss(model, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     args.milestones,
                                                     gamma=args.gamma)

    step = 0

    for epoch in range(args.epochs):
        step += 1
        optimizer.zero_grad()

        y_logits = model(x_train)
        neg_elbo, categorical_mean, kl, logsoftmax = loss(y_logits, y_onehot_train, step)

        pred = torch.argmax(logsoftmax, dim=1)
        neg_elbo.backward()

        nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)
        scheduler.step()
        optimizer.step()

        if epoch % args.draw_every == 0:
            print("epoch : {}".format(epoch))
            print("ELBO : {:.4f}\t categorical_mean: {:.4f}\t KL: {:.4f}".format(
                -neg_elbo.item(), categorical_mean.item(), kl.item()))

            draw_classification_results(x_train, pred, 'after_{}_epoch.png', args)

    with torch.no_grad():
        y_logits = model(x_train)
        _, _, _, logsoftmax = loss(y_logits, y_onehot_train, step)
        pred = torch.argmax(logsoftmax, dim=1)
        draw_classification_results(x_train, pred, 'end_train.png', args)

        y_logits = model(x_test)
        _, _, _, logsoftmax = loss(y_logits, y_onehot_test, step)
        pred = torch.argmax(logsoftmax, dim=1)
        draw_classification_results(x_test, pred, 'end_test.png', args)
