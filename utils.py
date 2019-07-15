import math

import matplotlib.pyplot as plt
import numpy as np
import torch


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
