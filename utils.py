import math
import os
import tqdm
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification, make_circles
from torchvision import datasets, transforms


def one_hot_encoding(tensor, n_classes, device):
    ohe = torch.LongTensor(tensor.size(0), n_classes).to(device)
    ohe.zero_()
    ohe.scatter_(1, tensor, 1)
    return ohe


def toy_results_plot_regression(data, data_generator, predictions=None,
                                name=None):
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

        plt.scatter(x, y_mean - full_std, color='b', alpha=0.2,
                    label='model 1-std')
        plt.scatter(x, y_mean + full_std, color='b', alpha=0.2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([-3, 2])
    plt.legend()
    plt.savefig(name)
    plt.close()


def get_predictions(data, model, args, mcvi=False):
    output = model(data)

    output_cov = output[1]
    output_mean = output[0]

    n = output_mean.size(0)

    if not mcvi:
        m = output_mean.size(1)

        out_cov = torch.reshape(output_cov, (n, m, m))
        out_mean = output_mean

        x = data.cpu().detach().numpy().squeeze()
        y = {'mean': out_mean.cpu().detach().numpy(),
             'cov': out_cov.cpu().detach().numpy()}
    else:
        out_mean = output_mean.unsqueeze(-1)
        out_cov = torch.zeros_like(out_mean).unsqueeze(-1)

        x = data.cpu().detach().numpy().squeeze()
        y = {'mean': out_mean.cpu().detach().numpy(),
             'cov': out_cov.cpu().detach().numpy()}
    return x, y


def generate_regression_data(args, sample_data, base_model):
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


def generate_classification_data(args):
    if args.dataset.strip().lower() == 'classification':
        n_informative = int(args.input_size * 0.8)
        x_train, y_train = make_classification(n_samples=args.data_size,
                                               n_features=args.input_size,
                                               n_informative=n_informative,
                                               n_redundant=args.input_size - n_informative,
                                               n_classes=args.n_classes)

        x_test, y_test = make_classification(n_samples=args.test_size,
                                             n_features=args.input_size,
                                             n_informative=n_informative,
                                             n_redundant=args.input_size - n_informative,
                                             n_classes=args.n_classes)
        n_classes = args.n_classes
    elif args.dataset.strip().lower() == 'circles':
        x_train, y_train = make_circles(n_samples=args.data_size)
        x_test, y_test = make_circles(n_samples=args.test_size)
        n_classes = 2

    x_train = torch.FloatTensor(x_train).to(args.device)
    y_train = torch.LongTensor(y_train).to(args.device).view(-1, 1)

    y_onehot_train = one_hot_encoding(y_train, n_classes, args.device)

    x_test = torch.FloatTensor(x_test).to(args.device)
    y_test = torch.LongTensor(y_test).to(args.device).view(-1, 1)

    y_onehot_test = one_hot_encoding(y_test, n_classes, args.device)

    return x_train, y_train, y_onehot_train, x_test, y_test, y_onehot_test


def draw_classification_results(data, prediction, name, args):
    """

    :param data: input to draw, should be 2D
    :param prediction: predicted class labels
    :param name: output file name
    :return:
    """

    x = data.detach().cpu().numpy()
    y = prediction.detach().cpu().numpy().squeeze()

    plt.figure(figsize=(10, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y)

    if args.dataset == "circles":
        path = 'pics/classification/circles'
    else:
        path = 'pics/classification/cls'

    if not os.path.exists(path):
        os.mkdir(path)

    filename = path + os.sep + name
    plt.savefig(filename)
    plt.close()


def load_mnist(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True)
    return train_loader, test_loader


def save_checkpoint(state, dir, filename):
    torch.save(state, os.path.join(dir, filename))


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)

    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    elbo = checkpoint['elbo']
    model.load_state_dict(checkpoint['state_dict'])
    return epoch, accuracy, elbo


def report(dir, epoch, elbo, cat_mean, kl, accuracy, test_acc_prob):
    message = "\nELBO : {:.4f}\t categorical_mean: {:.4f}\t KL: {:.4f}\n".format(
        elbo, cat_mean, kl)
    message += "train accuracy: {:.4f}\t".format(accuracy)
    message += "test_accuracy(probs): {:.4f}\t".format(test_acc_prob)
    print(message)

    message = "\nepoch: {}\n".format(epoch) + message
    with open(os.path.join(dir, 'report'), 'a') as f:
        print(message, file=f)


def prepare_directory(args):
    if args.checkpoint_dir == '':
        args.checkpoint_dir = os.path.join('checkpoints', 'last_expirement')
    else:
        args.checkpoint_dir = os.path.join('checkpoints', args.checkpoint_dir)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.system('rm -rf %s/*' % args.checkpoint_dir)

    print(args)
    with open(os.path.join(args.checkpoint_dir, 'hypers'), 'w') as f:
        print(args, file=f)


def mc_prediction(model, input, n_samples):
    logits = torch.stack([model(input)[0] for _ in range(n_samples)], dim=0)
    probs = F.softmax(logits, dim=-1)
    mean_probs = torch.mean(probs, dim=0)
    return mean_probs
