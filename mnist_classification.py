import argparse
import os

import numpy as np
import torch
import tqdm
from torch import nn

from bayesian_utils import classification_posterior, sample_softmax
from losses import ClassificationLoss
from models import LinearDVI, LeNetDVI, LinearVDO, LeNetVDO
from utils import load_mnist, save_checkpoint, report, prepare_directory, \
    mc_prediction, one_hot_encoding

np.random.seed(42)

EPS = 1e-6

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--arch', type=str, default="lenet")
parser.add_argument('--anneal_updates', type=int, default=20)
parser.add_argument('--warmup_updates', type=int, default=30)
parser.add_argument('--mcvi', action='store_true')
parser.add_argument('--mc_samples', default=10, type=int)
parser.add_argument('--clip_grad', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.5,
                    help='lr decrease rate in MultiStepLR scheduler')
parser.add_argument('--milestones', nargs='+', type=int,
                    default=[30, 50, 80, 95, 120])
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--use_samples', action='store_true',
                    help='use mc samples for determenistic probs on test stage.')
parser.add_argument('--swap_modes', action='store_true',
                    help="use different modes for train and test")
parser.add_argument('--var_network', action='store_true')
parser.add_argument('--n_var_layers', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='')

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = load_mnist(args)
    args.data_size = len(train_loader.dataset)

    prepare_directory(args)

    if args.arch.strip().lower() == "fc" and not args.var_network:
        model = LinearDVI(args).to(args.device)
    elif args.arch.strip().lower() == "fc" and args.var_network:
        model = LinearVDO(args).to(args.device)
    elif args.arch.strip().lower() == "lenet" and not args.var_network:
        model = LeNetDVI(args).to(args.device)
    else:
        model = LeNetVDO(args).to(args.device)

    print(model)

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

    use_det_on_train = not args.mcvi
    for epoch in range(args.epochs):
        model.train()
        model.set_flag('zero_mean', False)

        if use_det_on_train and args.swap_modes:
            model.set_flag('deterministic', True)
            args.mcvi = False
        elif args.swap_modes:
            model.set_flag('deterministic', False)
            args.mcvi = True

        print('\nepoch:', epoch)
        if scheduler is not None:
            scheduler.step()
        criterion.step()

        elbo, cat_mean, kls, accuracy = [], [], [], []
        for data, y_train in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            if args.arch == "fc":
                x_train = data.view(-1, 28 * 28).to(args.device)
            else:
                x_train = data.to(args.device)

            y_train = y_train.to(args.device)

            y_ohe = one_hot_encoding(y_train[:, None], 10, args.device)
            y_logits = model(x_train)

            print('*' * 50, end='\n\n')

            print('compute loss....')
            loss, categorical_mean, kl, logsoftmax = criterion(y_logits,
                                                               y_ohe)
            print('finished')

            pred = torch.argmax(logsoftmax, dim=1)
            print('backward...')
            loss.backward()
            print('finished')

            print('optimizer step...')
            if args.clip_grad > 0:
                nn.utils.clip_grad.clip_grad_value_(model.parameters(),
                                                    args.clip_grad)

            optimizer.step()
            print('finished')

            print('accuracy...')
            elbo.append(-loss.item())
            cat_mean.append(categorical_mean.item())
            kls.append(kl.item())
            accuracy.append(
                (torch.sum(torch.squeeze(pred) == torch.squeeze(y_train),
                           dtype=torch.float32) / args.batch_size).item())
            print('finished')

        elbo = np.mean(elbo)
        cat_mean = np.mean(cat_mean)
        kl = np.mean(kls)
        accuracy = np.mean(accuracy)

        print("\nTest prediction")
        if use_det_on_train and args.swap_modes:
            model.set_flag('deterministic', False)
            args.mcvi = True
        elif args.swap_modes:
            model.set_flag('deterministic', True)
            args.mcvi = False

        model.eval()

        test_acc_prob = []
        test_acc_log_prob = []
        with torch.no_grad():
            for data, y_test in tqdm.tqdm(test_loader):
                if args.arch == "fc":
                    x = data.view(-1, 28 * 28).to(args.device)
                else:
                    x = data.to(args.device)

                y = y_test.to(args.device)

                if args.mcvi:
                    probs = mc_prediction(model, x, args.mc_samples)
                elif args.use_samples:
                    activations = model(x)
                    probs = sample_softmax(activations,
                                           n_samples=args.mc_samples)
                else:
                    activations = model(x)
                    probs = classification_posterior(activations)

                pred = torch.argmax(probs, dim=1)
                test_acc_prob.append(
                    (torch.sum(torch.squeeze(pred) == torch.squeeze(y),
                               dtype=torch.float32) / args.test_batch_size).item())

            test_acc_prob = np.mean(test_acc_prob)

        zero_mean_acc = None
        if args.var_network:
            model.zero_mean()
            zero_mean_acc = []
            with torch.no_grad():
                for data, y_test in tqdm.tqdm(test_loader):
                    if args.arch == "fc":
                        x = data.view(-1, 28 * 28).to(args.device)
                    else:
                        x = data.to(args.device)

                    y = y_test.to(args.device)

                    if args.mcvi:
                        probs = mc_prediction(model, x, args.mc_samples)
                    elif args.use_samples:
                        activations = model(x)
                        probs = sample_softmax(activations,
                                               n_samples=args.mc_samples)
                    else:
                        activations = model(x)
                        probs = classification_posterior(activations)

                    pred = torch.argmax(probs, dim=1)
                    zero_mean_acc.append(
                        (torch.sum(torch.squeeze(pred) == torch.squeeze(y),
                                   dtype=torch.float32) / args.test_batch_size).item())

            model.zero_mean(False)
            zero_mean_acc = np.mean(zero_mean_acc)

        model.print_alphas()
        report(args.checkpoint_dir, epoch, elbo, cat_mean, kl, accuracy,
               test_acc_prob, zero_mean_acc)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'elbo': elbo,
            'train_accuracy': accuracy,
            'test_accuracy': test_acc_prob,
            'zero_mean_acc': zero_mean_acc,
            'kl': kl,
            'll': cat_mean
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
