import argparse
import os

import numpy as np
import torch
import tqdm
from torch import nn

from losses import ClassificationLoss, logsoftmax_mean, sample_softmax
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
        criterion.step()

        optimizer.zero_grad()

        elbo, cat_mean, kls, accuracy = [], [], [], []
        for data, y_train in tqdm.tqdm(train_loader):
            x_train = data.view(-1, 28 * 28).to(args.device)
            y_train = y_train.to(args.device)

            y_ohe = one_hot_encoding(y_train[:, None], 10, args.device)
            y_logits = model(x_train)

            loss, categorical_mean, kl, logsoftmax = criterion(y_logits,
                                                               y_ohe)
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

        test_acc_prob = []
        test_acc_log_prob = []
        with torch.no_grad():
            for data, y_test in test_loader:
                x = data.view(-1, 28 * 28).to(args.device)
                y = y_test.to(args.device)

                logits = model(x)

                probs = sample_softmax(logits, n_samples=args.n_samples)
                pred = torch.argmax(probs, dim=1)
                test_acc_prob.append(
                    (torch.sum(torch.squeeze(pred) == torch.squeeze(y),
                               dtype=torch.float32) / args.batch_size).item())

                log_probs = logsoftmax_mean(logits)
                pred = torch.argmax(log_probs, dim=1)
                test_acc_log_prob.append(
                    (torch.sum(torch.squeeze(pred) == torch.squeeze(y),
                               dtype=torch.float32) / args.batch_size).item())

            test_acc_prob = np.mean(test_acc_prob)
            test_acc_log_prob = np.mean(test_acc_log_prob)

        print(
            "ELBO : {:.4f}\t categorical_mean: {:.4f}\t KL: {:.4f}".format(
                elbo, cat_mean, kl))
        print(
            "train accuracy: {:.4f}\t test_accuracy (sample probs): {:.4f}\t test_accuracy (mean logprob): {:.4f}".format(
                accuracy,
                test_acc_prob, test_acc_log_prob))

        if epoch > int(args.epochs / 10):
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'elbo': elbo,
                'train_accuracy': accuracy,
                'test_accuracy (sample)': test_acc_prob,
                'test_accuracy (mean_logsoftmax)': test_acc_log_prob
            }

            save_checkpoint(state, True, 'checkpoints', 'epoch{}.pth.tar'.format(epoch))
            if test_acc_prob > best_test_acc:
                best_test_acc = test_acc_prob
                save_checkpoint(state, True, 'checkpoints', 'best.pth.tar')

        if epoch % 11 == 0 and epoch > 0:
            os.system('clear')
