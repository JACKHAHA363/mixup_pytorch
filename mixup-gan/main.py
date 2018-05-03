import os, sys
from tensorboardX import SummaryWriter
import argparse
import datetime
import ipdb
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import InfEightGaussiansGen, InfTwentyFiveGaussiansGen

######################CONFIG####################
MODE = 'gan-mixup'  # gan or gan-mixup
DATASET = '8gaussian'  # 8gaussians or 25gaussian,
DIM = 512  # Model dimensionality
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 128  # Batch size
ITERS = 1500  # how many generator iterations to train for
ALPHA = 0.8
PRINT_FREQ = 100
"""
The networks are trained for 20,000 mini-batches of size
128 using the Adam optimizer with default parameters, where the discriminator is trained for five
iterations before every generator iteration

This is unclear here. Does 20000 mini batches including the batch for discriminator, which is 5 times more?
"""
USE_CUDA = torch.cuda.is_available()
FOLDER = './' + MODE + '_' + DATASET

if os.path.exists(FOLDER):
    print('{} exists'.format(FOLDER))
    exit(0)


#==================Definition Start======================

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def prepare_D_batch(real_batch, fake_batch):
    """
    :param real_batch: [bsz, 2]. torch Tensor
    :param fake_batch: [bsz, 2]. torch Tensor
    :return:
        Non mixup:
        D_batch: [2*bsz, 2]. Variable
        D_targets: [2*bsz]. Variable

        mixup:
        D_batch: [2*bsz, 2]. Variable
        D_targets: [2*bsz]. Variable
    """
    if MODE == 'gan':
        D_batch = torch.cat([real_batch, fake_batch], dim=0)
        D_targets = torch.cat([
            torch.ones(BATCH_SIZE), torch.zeros(BATCH_SIZE)
        ], dim=0)
    elif MODE == 'gan-mixup':
        lam = np.random.beta(ALPHA, ALPHA)
        D_batch = lam * real_batch + (1-lam) * fake_batch
        D_targets = torch.ones(BATCH_SIZE)*lam

    if USE_CUDA:
        D_targets = D_targets.cuda()
    return Variable(D_batch), Variable(D_targets)


def plot_everything(real_data, fake_data, name):
    """
    :param real_data: np array
    :param fake_data: np array
    :param name: save as FOLDER/name.png
    """
    plt.figure()
    real_plt, = plt.plot(real_data[:,0], real_data[:,1], 'r.')
    fake_plt, = plt.plot(fake_data[:,0], fake_data[:,1], 'b.')
    plt.legend([real_plt, fake_plt], ["real", "fake"])
    plt.savefig(os.path.join(FOLDER, name+'.png'))
    plt.close()

# ==================Definition End======================

if DATASET == '8gaussian':
    data_gen = InfEightGaussiansGen(BATCH_SIZE)
elif DATASET == '25gaussian':
    data_gen = InfTwentyFiveGaussiansGen(BATCH_SIZE)
else:
    print("No dataset!")

netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)

if USE_CUDA:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4)
optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

writer = SummaryWriter(log_dir=os.path.join(FOLDER, 'logs'))

for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):
        # prepare real and fake batch
        _data = data_gen.__next__()
        real_data = torch.FloatTensor(_data)
        if USE_CUDA:
            real_data = real_data.cuda()

        # train with fake
        noise = torch.randn(BATCH_SIZE, 2)
        if USE_CUDA:
            noise = noise.cuda()
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake_data = netG(noisev).data

        D_batch, D_targets = prepare_D_batch(real_data, fake_data)
        logits = netD(D_batch)
        D_cost = criterion(logits, D_targets)

        netD.zero_grad()
        D_cost.backward()
        optimizerD.step()


    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation


    noise = torch.randn(BATCH_SIZE, 2)
    if USE_CUDA:
        noise = noise.cuda()
    noisev = autograd.Variable(noise)

    if MODE == 'gan':
        fake_logits = netD(netG(noisev))
        fake_targets = torch.zeros(BATCH_SIZE)
        if USE_CUDA:
            fake_targets = fake_targets.cuda()
        fake_targets = Variable(fake_targets)
        G_cost = -criterion(fake_logits, fake_targets)
    elif MODE == 'gan-mixup':
        fake_data_v = netG(noisev)
        lam = np.random.beta(ALPHA, ALPHA)
        G_batch = lam * Variable(real_data) + (1-lam) * fake_data_v
        G_targets = torch.ones(BATCH_SIZE)*lam
        if USE_CUDA:
            G_targets = G_targets.cuda()
        G_cost = -criterion(netD(G_batch), Variable(G_targets))

    netG.zero_grad()
    G_cost.backward()
    optimizerG.step()

    if (iteration+1) % PRINT_FREQ == 0:
        print("at iteration {iteration}\tD_cost:{D_cost.data[0]}\tG_cost:{G_cost.data[0]}".format(
            iteration=iteration+1, D_cost=D_cost, G_cost=G_cost
        ))
        log_dict = {
            "D_cost": D_cost, "G_cost": G_cost
        }
        for key in log_dict.keys():
            writer.add_scalar(key, log_dict[key], iteration+1)

        # plot stuff
        plot_everything(real_data.cpu().numpy(), fake_data.cpu().numpy(), str(iteration+1))