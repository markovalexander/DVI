import argparse

import numpy as np
import torch
from torch import nn

from losses import ClassificationLoss
from models import LinearDVI, LeNetDVI
from utils import load_mnist, one_hot_encoding, get_statistics

import tqdm

np.random.seed(42)

EPS = 1e-6

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mcvi', action='store_true')
parser.add_argument('--arch', type=str, default="fc")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--anneal_updates', type=int, default=1000)
parser.add_argument('--warmup_updates', type=int, default=14000)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--gamma', type=float, default=0.5,
                    help='lr decrease rate in MultiStepLR scheduler')
parser.add_argument('--epochs', type=int, default=23000)
parser.add_argument('--milestones', nargs='+', type=int,
                    default=[3000, 5000, 9000, 13000])
parser.add_argument('--mc_samples', default=1, type=int)
parser.add_argument('--report_every', type=int, default=100)

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    print(args)

    train_loader, test_loader = load_mnist(args)

    if args.arch.strip().lower() == "fc":
        model = LinearDVI(args).to(args.device)
    else:
        model = LeNetDVI(args).to(args.device)

    criterion = ClassificationLoss(model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     args.milestones,
                                                     gamma=args.gamma)

    step = 0

    for epoch in range(args.epochs):
        step += 1
        optimizer.zero_grad()

        for data, y_train in tqdm.tqdm(train_loader):
            x_train = data.view(-1, 28 * 28).to(args.device)
            y_train = y_train.to(args.device)

            y_ohe = one_hot_encoding(y_train[:, None], 10, args.device)
            y_logits = model(x_train)

            loss, categorical_mean, kl, logsoftmax = criterion(y_logits,
                                                               y_ohe, step)
            pred = torch.argmax(logsoftmax, dim=1)
            loss.backward()

            nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)
            scheduler.step()
            optimizer.step()

        if epoch % args.report_every == 0:
            elbo, cat_mean, kl, accuracy = get_statistics(model, criterion,
                                                          train_loader, step,
                                                          args)
            print("epoch : {}".format(epoch))
            print(
                "ELBO : {:.4f}\t categorical_mean: {:.4f}\t KL: {:.4f}".format(
                    elbo, cat_mean, kl))
            print("train accuracy: {:.4f}".format(accuracy))
