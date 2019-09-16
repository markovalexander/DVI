import argparse
from time import time

import numpy as np

from core.logger import Logger
from core.losses import ClassificationLoss
from core.models import *
from core.utils import load_mnist, one_hot_encoding, evaluate, pred2acc

np.random.seed(42)

EPS = 1e-6

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--anneal_updates', type=int, default=1)
parser.add_argument('--warmup_updates', type=int, default=0)
parser.add_argument('--mcvi', action='store_true')
parser.add_argument('--mc_samples', default=20, type=int)
parser.add_argument('--clip_grad', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--zm', action='store_true')
parser.add_argument('--no_mc', action='store_true')
parser.add_argument('--use_samples', action='store_true')
parser.add_argument('--reshape', action='store_true', default=False)


fmt = {'kl': '3.3e',
       'tr_elbo': '3.3e',
       'tr_acc': '.4f',
       'tr_ll': '.3f',
       'te_acc_dvi': '.4f',
       'te_acc_mcvi': '.4f',
       'te_acc_samples': '.4f',
       'te_acc_dvi_zm': '.4f',
       'te_acc_mcvi_zm': '.4f',
       'tr_time': '.3f',
       'te_time_dvi': '.3f',
       'te_time_mcvi': '.3f'}

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device(
        'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = LeNetDVI(args).to(args.device)

    args.batch_size, args.test_batch_size = 32, 32
    train_loader, test_loader = load_mnist(args)
    args.data_size = len(train_loader.dataset)

    logger = Logger('lenet-deterministic', fmt=fmt)
    logger.print(args)
    logger.print(model)

    criterion = ClassificationLoss(model, args)
    optimizer = torch.optim.Adam([p for p in model.parameters()
                                  if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        t0 = time()

        model.train()
        criterion.step()

        elbo, cat_mean, kls, accuracy = [], [], [], []
        for data, y_train in train_loader:
            optimizer.zero_grad()

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
            accuracy.append(pred2acc(pred, y_train, args.batch_size))

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

        if not args.no_mc:
            t_mc = time()
            test_acc_mcvi = evaluate(model, test_loader, mode='mcvi', args=args)
            t_mc = time() - t_mc
            logger.add(epoch, te_acc_mcvi=test_acc_mcvi, te_time_mcvi=t_mc)

        test_acc_samples = evaluate(model, test_loader, mode='samples_dvi',
                                    args=args)
        logger.add(epoch, te_acc_dvi=test_acc_dvi,
                   te_acc_samples=test_acc_samples, te_time_dvi=t_dvi)

        logger.iter_info()
        logger.save(silent=True)
        torch.save(model.state_dict(), logger.checkpoint)
    logger.save()
