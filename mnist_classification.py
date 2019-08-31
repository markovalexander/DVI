import argparse
from time import time

import numpy as np

from logger import Logger
from losses import ClassificationLoss
from models import *
from utils import load_mnist, one_hot_encoding, evaluate

np.random.seed(42)

EPS = 1e-6

parser = argparse.ArgumentParser()

parser.add_argument('--var_conv', action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--arch', type=str, default="lenet")
parser.add_argument('--anneal_updates', type=int, default=20)
parser.add_argument('--warmup_updates', type=int, default=30)
parser.add_argument('--mcvi', action='store_true')
parser.add_argument('--mc_samples', default=20, type=int)
parser.add_argument('--clip_grad', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.5,
                    help='lr decrease rate in MultiStepLR scheduler')
parser.add_argument('--milestones', nargs='+', type=int, default=[])
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--var_network', action='store_true')
parser.add_argument('--vdo', action='store_true')
parser.add_argument('--n_layers', type=int, default=1,
                    help='number of variance or VDO layers for \
                    this architectures')
parser.add_argument('--nonlinearity', type=str, default='relu')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--zm', action='store_true')
parser.add_argument('--no_mc', action='store_true')
parser.add_argument('--use_samples', action='store_true')

fmt = {'kl': '3.3e',
       'tr_elbo': '3.3e',
       'tr_acc': '.4f',
       'tr_ll': '.3f',
       'te_acc_dvi': '.4f',
       'te_acc_mcvi': '.4f',
       'te_acc_samples': '.4f',
       'te_acc_dvi_zero_mean': '.4f',
       'te_acc_mcvi_zero_mean': '.4f',
       'tr_time': '.3f',
       'te_time_dvi': '.3f',
       'te_time_mcvi': '.3f'}


if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = load_mnist(args)
    args.data_size = len(train_loader.dataset)

    if args.arch.strip().lower() == "fc":
        if args.vdo:
            model = LinearVDO(args).to(args.device)
        else:
            model = LinearDVI(args).to(args.device)
    elif args.arch.strip().lower() == "lenet":
        if args.zm:
            model = LeNetFullVariance(args).to(args.device)
        elif args.var_network:
            model = LeNetVariance(args).to(args.device)
        elif args.vdo:
            model = LeNetVDO(args).to(args.device)
        else:
            model = LeNetDVI(args).to(args.device)

    logger_name = args.arch + ('-vdo' if args.vdo else '') \
                  + ('-vn' if args.var_network else '') \
                  + ('-dvi' if not (args.vdo or args.var_network) else '') \
                  + ('-{}'.format(args.name) if len(args.name) > 0 else '')

    for layer in model.children():
        i = 0
        if hasattr(layer, 'log_alpha'):
            fmt.update({'{}log_alpha'.format(i + 1): '2.2e'})
            i += 1

    logger = Logger(logger_name, fmt=fmt)
    logger.print(args)
    logger.print(model)

    criterion = ClassificationLoss(model, args)
    optimizer = torch.optim.Adam([p for p in model.parameters()
                                  if p.requires_grad], lr=args.lr)

    if len(args.milestones) > 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         args.milestones,
                                                         gamma=args.gamma)
    else:
        scheduler = None

    for epoch in range(args.epochs):
        t0 = time()

        model.train()
        model.set_flag('zero_mean', False)

        if scheduler is not None:
            scheduler.step()
        criterion.step()

        elbo, cat_mean, kls, accuracy = [], [], [], []
        for data, y_train in train_loader:
            optimizer.zero_grad()

            if args.arch == "fc":
                x_train = data.view(-1, 28 * 28).to(args.device)
            else:
                x_train = data.to(args.device)

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
            kls.append(kl.item() if isinstance(kl, torch.Tensor) else kl)
            accuracy.append(
                (torch.sum(torch.squeeze(pred) == torch.squeeze(y_train),
                           dtype=torch.float32) / args.batch_size).item())

        t1 = time() - t0
        elbo = np.mean(elbo)
        cat_mean = np.mean(cat_mean)
        kl = np.mean(kls)
        accuracy = np.mean(accuracy)
        logger.add(epoch, kl=kl, tr_elbo=elbo, tr_acc=accuracy, tr_ll=cat_mean,
                   tr_time=t1)

        model.eval()
        t_dvi = time()
        test_acc_dvi = evaluate(model, test_loader, mode='dvi', args=args)
        t_dvi = time() - t_dvi

        if args.no_mc:
            t_mc = 0
            test_acc_mcvi = 0
        else:
            t_mc = time()
            test_acc_mcvi = evaluate(model, test_loader, mode='mcvi', args=args)
            t_mc = time() - t_mc

        test_acc_samples = evaluate(model, test_loader, mode='samples_dvi',
                                    args=args)

        logger.add(epoch, te_acc_dvi=test_acc_dvi, te_acc_mcvi=test_acc_mcvi,
                   te_acc_samples=test_acc_samples, te_time_dvi=t_dvi,
                   te_time_mcvi=t_mc)

        if isinstance(model, LeNetVDO) or isinstance(model, LinearVDO):
            test_acc_zero_mean_dvi = evaluate(model, test_loader, mode='dvi',
                                              args=args, zero_mean=True)
            test_acc_zero_mean_mcvi = evaluate(model, test_loader, mode='mcvi',
                                               args=args, zero_mean=True)

            logger.add(epoch, te_acc_dvi_zero_mean=test_acc_zero_mean_dvi,
                       te_acc_mcvi_zero_mean=test_acc_zero_mean_mcvi)
            i = 0
            alphas = {}
            for layer in model.children():
                if hasattr(layer, 'log_alpha'):
                    alphas.update(
                        {'{}_log_alpha'.format(i + 1): layer.log_alpha.item()})
                    i += 1
            logger.add(epoch, **alphas)

        logger.iter_info()
        logger.save(silent=True)
        torch.save(model.state_dict(), logger.checkpoint)
    logger.save()
