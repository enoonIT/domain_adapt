'''
This is a PyTorch implementation of 'Domain-Adversarial Training of 
Neural Networks' by Yaroslav Ganin et al. (2016).

The DANN model uses the adversarial learning paradigm to force a 
classifier to only learn features that exist in both domains. This
enables a classifier trained on the source domain to generalize to 
the target domain.

This is achieved with the 'gradient reversal' layer to form
a domain invariant feature embedding which can be used with the 
same CNN.

This example uses MNIST as source dataset and USPS or MNIST-M 
as target datasets.

Author: Daniel Bartolomé Rojas (d.bartolome.r@gmail.com)
'''
import argparse

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.backends import cudnn

from evaluation import eval_clf
from logger import Logger
from models import DANN, DANN_deco
from data_handling import batch_generator, data_loader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2)
    # parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_src_epochs', default=15, type=int,
                        help="number of epochs to pre-train source only model")
    parser.add_argument('--epochs', default=30, type=int)
    # parser.add_argument('--DANN_weight', default=1.0, type=float)
    parser.add_argument('--use_deco', action="store_true", help="If true use deco architecture")
    parser.add_argument('--suffix', help="Will be added to end of name", default="")
    parser.add_argument('--source', default="mnist")
    parser.add_argument('--target', default="usps")
    return parser.parse_args()


def get_name(args):
    name = "lr:%g_batchSize:%d_epochs:%d" % (args.lr, args.batch_size, args.epochs)
    if args.use_deco:
        name += "_deco"
    return name + args.suffix + "_%d" % (time.time() % 100)


cuda = True
cudnn.benchmark = True
args = get_args()
run_name = get_name(args)
logger = Logger("../logs/" + run_name)
# parameters
learning_rate = args.lr  # learning rate
num_epochs = args.epochs  # number of epochs to train models
num_src_epochs = args.num_src_epochs  # number of epochs to pre-train source only model
batch_size = args.batch_size  # size of image sample per epoch
source_data = args.source  # mnist / mnist_m / usps
target_data = args.target  # mnist / mnist_m / usps
input_ch = 1  # 1 for usps, 3 for mnistm
if target_data is "mnist_m":
    input_ch = 3

# instantiate the models
if args.use_deco:
    f_ext, d_clf, c_clf = DANN_deco(input_ch)
else:
    f_ext, d_clf, c_clf = DANN(input_ch)

# set loss functions
d_crit = nn.BCELoss()  # binary crossentropy
c_crit = nn.MSELoss()  # mean squared error

if cuda:
    f_ext.cuda()
    d_clf.cuda()
    c_clf.cuda()
    d_crit.cuda()
    c_crit.cuda()

# set optimizers
d_optimizer = optim.SGD(d_clf.parameters(), lr=learning_rate, momentum=0.9)
c_optimizer = optim.SGD(c_clf.parameters(), lr=learning_rate, momentum=0.9)
f_optimizer = optim.SGD(f_ext.parameters(), lr=learning_rate, momentum=0.9)

# load source domain dataset
(Xs_train, ys_train), (Xs_test, ys_test) = data_loader(source_data)

# same lengths as USPS dataset
(Xs_train, ys_train), (Xs_test, ys_test) = (Xs_train[:7291], ys_train[:7291]), (Xs_test[:2007], ys_test[:2007])
# # concat MNIST images as channels to match number of MNIST-M channels
# Xs_train = np.concatenate([Xs_train, Xs_train, Xs_train], axis=1)
# Xs_test = np.concatenate([Xs_test, Xs_test, Xs_test], axis=1)

# load target domain dataset
(Xt_train, yt_train), (Xt_test, yt_test) = data_loader(target_data)

# init necessary objects
num_steps = num_epochs * (Xs_train.shape[0] / batch_size)
yd = Variable(
    torch.from_numpy(np.hstack([np.repeat(1, int(batch_size / 2)), np.repeat(0, int(batch_size / 2))]).reshape(50, 1)))
if cuda:
    yd = yd.cuda()
j = 0

# pre-train source only model
print('\nPre-training source-only model..')
for i in range(num_src_epochs):
    source_gen = batch_generator(int(batch_size / 2), Xs_train, ys_train)

    # iterate over batches
    for (xs, ys, _) in source_gen:

        # exit when batch size mismatch
        if len(xs) != batch_size / 2:
            continue

        # reset gradients
        f_ext.zero_grad()
        c_clf.zero_grad()

        # calculate class_classifier predictions
        if cuda:
            xs = xs.cuda()
            ys = ys.cuda()
        c_out = c_clf(f_ext(xs).view(int(batch_size / 2), -1))

        # optimize feature_extractor and class_classifier with output
        f_c_loss = c_crit(c_out, ys.float())
        f_c_loss.backward(retain_graph=True)
        c_optimizer.step()
        f_optimizer.step()

        # print batch statistics
        print('\rEpoch {}       - loss: {}'.format(i + 1, format(f_c_loss.cpu().data[0], '.4f')), end='')

    # print epoch statistics    
    s_acc = eval_clf(c_clf, f_ext, Xs_test, ys_test, 1000)
    print(' - val_acc: {}'.format(format(s_acc, '.4f')))

# print target accuracy with source model
t_acc = eval_clf(c_clf, f_ext, Xt_test, yt_test, 1000)
print('\nTarget accuracy with source model: {}\n'.format(format(t_acc, '.4f')))

# train DANN model
print('Training DANN model..')
k = 0
for i in range(num_epochs):
    source_gen = batch_generator(int(batch_size / 2), Xs_train, ys_train)
    target_gen = batch_generator(int(batch_size / 2), Xt_train, None)

    # iterate over batches
    for (xs, ys, _) in source_gen:

        # update lambda and learning rate as suggested in the paper
        p = float(j) / num_steps
        lambd = round(2. / (1. + np.exp(-10. * p)) - 1, 3)
        lr = 0.01 / (1. + 10 * p) ** 0.75
        d_clf.set_lambda(lambd)
        d_optimizer.lr = lr
        c_optimizer.lr = lr
        f_optimizer.lr = lr
        j += 1

        # get next target batch
        xt, _ = next(target_gen)

        # exit when batch size mismatch
        if len(xs) + len(xt) != batch_size:
            continue

        # concatenate source and target batch
        x = torch.cat([xs, xt], 0)

        # 1) train feature_extractor and class_classifier on source batch
        # reset gradients
        f_ext.zero_grad()
        c_clf.zero_grad()

        if cuda:
            x = x.cuda()
            ys = ys.cuda()
        # calculate class_classifier predictions on batch xs
        c_out = c_clf(f_ext(xs).view(int(batch_size / 2), -1))

        # optimize feature_extractor and class_classifier on output
        f_c_loss = c_crit(c_out, ys.float())
        f_c_loss.backward(retain_graph=True)
        c_optimizer.step()
        f_optimizer.step()

        # 2) train feature_extractor and domain_classifier on full batch x
        # reset gradients
        f_ext.zero_grad()
        d_clf.zero_grad()

        # calculate domain_classifier predictions on batch x
        d_out = d_clf(f_ext(x).view(batch_size, -1))

        # use normal gradients to optimize domain_classifier
        f_d_loss = d_crit(d_out, yd.float())
        f_d_loss.backward(retain_graph=True)
        d_optimizer.step()
        f_optimizer.step()

        # print batch statistics
        print('\rEpoch         - d_loss: {} - c_loss: {}'.format(format(f_d_loss.cpu().data[0], '.4f'),
                                                                 format(f_c_loss.cpu().data[0], '.4f')), end='')
        if (k % 30) is 0:
            logger.scalar_summary("loss/source", f_c_loss.cpu().data[0], k)
            logger.scalar_summary("loss/domain", f_d_loss.cpu().data[0], k)
        k += 1

        # print epoch statistics
    t_acc = eval_clf(c_clf, f_ext, Xt_test, yt_test, 1000)
    s_acc = eval_clf(c_clf, f_ext, Xs_test, ys_test, 1000)
    logger.scalar_summary("acc/source", s_acc, k)
    logger.scalar_summary("acc/target", t_acc, k)
    print('\nTarget_acc: {} - source_acc: {}'.format(format(t_acc, '.4f'), format(s_acc, '.4f')))
