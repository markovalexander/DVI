import argparse
import os

import numpy as np
import torch
import tqdm
from torch import nn

from losses import ClassificationLoss
from models import LinearDVI, LeNetDVI
from utils import load_mnist, one_hot_encoding, save_checkpoint

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
parser.add_argument('--clip_grad', type=float, default=0.1)

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    os.system("clear")

    train_loader, test_loader = load_mnist(args)
    args.data_size = len(train_loader.dataset)
    print(args)

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
    best_test_acc = - 10 ** 9

    for epoch in range(args.epochs):
        print("epoch : {}".format(epoch))
        scheduler.step()
        step += 1
        optimizer.zero_grad()

        elbo, cat_mean, kls, accuracy = [], [], [], []
        for data, y_train in tqdm.tqdm(train_loader):
            x_train = data.view(-1, 28 * 28).to(args.device)
            y_train = y_train.to(args.device)

            y_ohe = one_hot_encoding(y_train[:, None], 10, args.device)
            y_logits = model(x_train)

            loss, categorical_mean, kl, logsoftmax = criterion(y_logits,
                                                               y_ohe, step)
            pred = torch.argmax(logsoftmax, dim=1)
            loss.backward()

            if args.clip_grad > 0:
                nn.utils.clip_grad.clip_grad_value_(model.parameters(),
                                                    args.clip_grad)

            optimizer.step()

            elbo.append(-loss.item())
            cat_mean.append(categorical_mean.item())
            kls.append(kl.item())
            accuracy.append(
                (torch.sum(torch.squeeze(pred) == torch.squeeze(y_train),
                           dtype=torch.float32) / args.batch_size).item())

        elbo = np.mean(elbo)
        cat_mean = np.mean(cat_mean)
        kl = np.mean(kls)
        accuracy = np.mean(accuracy)

        test_acc = []
        with torch.no_grad():
            for data, y_test in test_loader:
                x = data.view(-1, 28 * 28).to(args.device)
                y = y_test.to(args.device)

                y_ohe = one_hot_encoding(y[:, None], 10, args.device)
                y_logits = model(x)

                _, _, _, logsoftmax = criterion(y_logits,
                                                y_ohe, step)
                pred = torch.argmax(logsoftmax, dim=1)
                test_acc.append((torch.sum(torch.squeeze(pred) == torch.squeeze(y),
                           dtype=torch.float32) / args.batch_size).item())
            test_acc = np.mean(test_acc)

        print(
            "ELBO : {:.4f}\t categorical_mean: {:.4f}\t KL: {:.4f}".format(
                elbo, cat_mean, kl))
        print("train accuracy: {:.4f}\t test_accuracy: {:.4f}".format(accuracy, test_acc))

        if epoch > int(args.epochs / 10):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'elbo': elbo,
                'train_accuracy': accuracy,
                'test_accuracy': test_acc
            }, test_acc > best_test_acc, 'checkpoints', 'best_mnist.pth.tar')
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        if epoch % 11 == 0 and epoch > 0:
            os.system('clear')
