import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from bayesian_utils import classification_posterior
from losses import ClassificationLoss, logsoftmax_mean, sample_softmax
from models import LinearDVI, LeNetDVI
from utils import load_mnist, save_checkpoint, report, prepare_directory, \
    one_hot_encoding

np.random.seed(42)

EPS = 1e-6

# TODO: batch_size поменьше

# TODO: для предсказания занулить дисперсии и использовать мат.ожидания как веса
# TODO: (если это плохо работает, то это индикатор оч большой дисперсии)

# TODO: посмотреть галазами на сэмплы внутри слоев (после того как все научится)

# TODO: dvi_train_mcvi_test

# TODO: 1 слой а не 3, для линейной модели, и с склеарном сравнить линейным

# TODO: попрофилировать

# TODO: better device argument

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mcvi', action='store_true')
parser.add_argument('--arch', type=str, default="fc")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--anneal_updates', type=int, default=1000)
parser.add_argument('--warmup_updates', type=int, default=14000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.5,
                    help='lr decrease rate in MultiStepLR scheduler')
parser.add_argument('--epochs', type=int, default=23000)
parser.add_argument('--milestones', nargs='+', type=int, default=[])
parser.add_argument('--mc_samples', default=1, type=int)
parser.add_argument('--clip_grad', type=float, default=0.1)
parser.add_argument('--checkpoint_dir', type=str, default='')
parser.add_argument('--use_samples', action='store_true',
                    help='use mc samples for determenistic probs on test stage.')

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = load_mnist(args)
    args.data_size = len(train_loader.dataset)

    prepare_directory(args)

    if args.arch.strip().lower() == "fc":
        model = LinearDVI(args).to(args.device)
    else:
        model = LeNetDVI(args).to(args.device)

    criterion = ClassificationLoss(model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if len(args.milestones) > 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         args.milestones,
                                                         gamma=args.gamma)
    else:
        scheduler = None

    best_epoch = 0
    best_test_acc = - 10 ** 9

    for epoch in range(args.epochs):
        print('\nepoch:', epoch)
        if scheduler is not None:
            scheduler.step()
        criterion.step()

        elbo, cat_mean, kls, accuracy = [], [], [], []
        for data, y_train in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

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

                if args.mcvi:
                    probs = F.softmax(logits[0], dim=1)
                elif args.use_samples:
                    probs = sample_softmax(logits, n_samples=args.mc_samples)
                else:
                    probs = classification_posterior(logits[0], logits[1])

                pred = torch.argmax(probs, dim=1)
                test_acc_prob.append(
                    (torch.sum(torch.squeeze(pred) == torch.squeeze(y),
                               dtype=torch.float32) / args.batch_size).item())

                if args.mcvi:
                    log_probs = F.log_softmax(logits[0], dim=1)
                else:
                    log_probs = logsoftmax_mean(logits)
                pred = torch.argmax(log_probs, dim=1)
                test_acc_log_prob.append(
                    (torch.sum(torch.squeeze(pred) == torch.squeeze(y),
                               dtype=torch.float32) / args.batch_size).item())

            test_acc_prob = np.mean(test_acc_prob)
            test_acc_log_prob = np.mean(test_acc_log_prob)

        report(args.checkpoint_dir, epoch, elbo, cat_mean, kl, accuracy,
               test_acc_prob, test_acc_log_prob)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'elbo': elbo,
            'train_accuracy': accuracy,
            'test_accuracy (probs)': test_acc_prob,
            'test_accuracy (mean_logsoftmax)': test_acc_log_prob
        }

        save_checkpoint(state, args.checkpoint_dir,
                        'epoch{}.pth.tar'.format(epoch))
        save_checkpoint(state, args.checkpoint_dir,
                        'last.pth.tar')
        if test_acc_prob > best_test_acc:
            best_test_acc = test_acc_prob
            best_epoch = epoch
            print("=> Saving a new best\n")
            save_checkpoint(state, args.checkpoint_dir, 'best.pth.tar')

    with open(os.path.join(args.checkpoint_dir, 'report'), 'a') as f:
        print('\nBest Accuracy: {:.4f}\tBest epoch: {}'.format(best_test_acc,
                                                               best_epoch))
