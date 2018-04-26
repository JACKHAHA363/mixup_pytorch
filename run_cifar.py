from src.util.load_data import load_cifar10_data

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os, copy
import numpy as np

use_cuda = torch.cuda.is_available()