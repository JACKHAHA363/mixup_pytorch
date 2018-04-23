import os, sys
from tensorboardX import SummaryWriter
sys.path.append(os.getcwd())
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
import math

###################Config#################################3
DIM = 512
OUTDIR = './output'
BATCH_SIZE = 256
USE_INTERPOLATION = True
USE_CUDA = torch.cuda.is_available()
LAMBDA = 0.5 # gradient penalty weight

ITERS = 10000 # total number of training iterations
PRINT_FREQ = 50 # print frequency
CRITIC_ITERS = 2 # critic iter per generator iteration
GAN_MODE = 'minimax' # 'minimax' or 'wgan' or 'nsgan'

################## Model Definition #######################
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

################ Utilities ##########################

def calc_gradient_penalty(netD, real_data, fake_data):
    """
    :param netD: The discriminator
    :param real_data: [bsz, 2]
    :param fake_data: [bsz, 2]
    :return: mean gradient penalty
    """
    alpha = torch.zeros(BATCH_SIZE, 1)
    if not USE_INTERPOLATION:
        alpha = torch.bernoulli(alpha)
    else:
        alpha.uniform_(0, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if USE_CUDA else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if USE_CUDA:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if USE_CUDA else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 0) ** 2).mean() * LAMBDA
    return gradient_penalty

def logmeanexp(x):
    """
    :param x: [bsz, num_features]
    :return: [bsz]
    """
    # [batch_size, 1]
    max_logits = torch.max(x, dim=1, keepdim=True)[0]

    # [batch_size]
    sum_exp = torch.sum(torch.exp(x - max_logits), dim=1, keepdim=True)

    return (max_logits + torch.log(sum_exp) - math.log(x.size(1))).squeeze(1)

def plot_everything(real_data, fake_data, netD):
    """
    :param real_data: [N, 2]. np array
    :param fake_data: [N, 2]. np array
    :param netD: a nn module
    :return: plot the real data, fake data and net D decision boundary
    """
    plt.figure()
    plt_real, = plt.plot(real_data[:, 0], real_data[:, 1], 'bo')
    plt_fake, = plt.plot(fake_data[:, 0], fake_data[:, 1], 'rx')
    plt.legend([plt_real, plt_fake], ['real', 'fake'])

    # Set min and max values and give it some padding
    x_min = -5
    x_max = 5
    y_min = -5
    y_max = 5
    h = 0.03
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    inputs = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], 1)
    if USE_CUDA:
        inputs = torch.FloatTensor(inputs).cuda()
    else:
        inputs = torch.FloatTensor(inputs)

    # Predict the function value for the whole gid
    preds = netD.forward(Variable(inputs, volatile=True))
    #preds = torch.ceil(F.sigmoid(preds) - 0.5)
    preds = preds.data.cpu().numpy()
    preds = preds.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, preds, cmap=plt.cm.Spectral)


# ==================Definition End======================

#netG = Generator()
#netD = Discriminator()
#netD.apply(weights_init)
#netG.apply(weights_init)
#print(netG)
#print(netD)
#
#if USE_CUDA:
#    netD = netD.cuda()
#    netG = netG.cuda()
#
#optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
#optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
#
#writer = SummaryWriter(log_dir=os.path.join(OUTDIR, 'logs'))
#
#for iteration in range(ITERS):
#    ############################
#    # (1) Update D network
#    ###########################
#    for p in netD.parameters():  # reset requires_grad
#        p.requires_grad = True  # they are set to False below in netG update
#
#    for iter_d in range(CRITIC_ITERS):
#        _data = hierachy_data_gen(BATCH_SIZE)
#        real_data = torch.FloatTensor(_data)
#        if USE_CUDA:
#            real_data = real_data.cuda()
#        real_data_v = autograd.Variable(real_data)
#
#        netD.zero_grad()
#
#        # train with real
#        D_real = netD(real_data_v)
#        if GAN_MODE == 'minimax' or GAN_MODE == 'nsgan':
#            real_cost = -torch.log(F.sigmoid(D_real))
#        elif GAN_MODE == 'wgan':
#            real_cost = -D_real
#        else:
#            raise NotImplementedError
#        real_cost = real_cost.mean()
#        real_cost.backward()
#
#        # train with fake
#        noise = torch.randn(BATCH_SIZE, 2)
#        if USE_CUDA:
#            noise = noise.cuda()
#        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
#        fake_data = netG(noisev).data
#        fake_data_v = autograd.Variable(fake_data)
#        D_fake = netD(fake_data_v)
#        if GAN_MODE == 'minimax' or GAN_MODE == 'nsgan':
#            fake_cost = -torch.log(1 - F.sigmoid(D_fake))
#        elif GAN_MODE == 'wgan':
#            fake_cost = D_fake
#        else:
#            raise NotImplementedError
#        fake_cost = fake_cost.mean()
#        fake_cost.backward()
#
#        # train with gradient penalty
#        gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data)
#        gradient_penalty.backward()
#
#        D_cost = real_cost + fake_cost + gradient_penalty
#        if GAN_MODE == 'wgan':
#            Wasserstein_D = (D_real - D_fake).mean()
#        optimizerD.step()
#
#
#    ############################
#    # (2) Update G network
#    ############################
#    for p in netD.parameters():
#        p.requires_grad = False  # to avoid computation
#    netG.zero_grad()
#
#    noise = torch.randn(BATCH_SIZE, 2)
#    if USE_CUDA:
#        noise = noise.cuda()
#    noisev = autograd.Variable(noise)
#    D_G_z = netD(netG(noisev))
#    if GAN_MODE == 'minimax':
#        G_cost = torch.log(1 - F.sigmoid(D_G_z))
#    elif GAN_MODE == 'wgan':
#        G_cost = -D_G_z
#    elif GAN_MODE == 'nsgan':
#        G_cost = -torch.log(F.sigmoid(D_G_z))
#    G_cost = G_cost.mean()
#    G_cost.backward()
#
#    optimizerG.step()
#
#    # Write logs and save samples
#    if (iteration+1) % PRINT_FREQ == 0:
#        info = {
#            'cost_gen': G_cost,
#            'cost_disc': D_cost,
#            }
#        for tag, value in info.items():
#            writer.add_scalar(tag, value, iteration+1)
#        plot_everything(real_data.cpu().numpy(), fake_data.cpu().numpy(), netD)
#
#        plt.savefig(os.path.join(OUTDIR, 'tmp.png'))
#        plt.close()
#
#        tmpim = plt.imread(os.path.join(OUTDIR, 'tmp.png'))
#        writer.add_image('training', tmpim, iteration+1)
#
#        print('at iter {} cost_gen {} cost disc {}'.format(iteration+1, G_cost.data[0], D_cost.data[0]))
