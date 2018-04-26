from src.util.load_data import load_cifar10_data
from src.util.util import mixup_data_and_target

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os, copy
import numpy as np

use_cuda = torch.cuda.is_available()

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
						help='Learning rate')
	parser.add_argument('-s', '--seed', default=123, type=int, help='Random seed')
	parser.add_argument('--batch_size', default=128, type=int,
						help='Mini-batch size')
	parser.add_argument('--epochs', default=600, type=int, help='Number of epochs')
	parser.add_argument('-a', '--alpha', default=0.6, type=float,
						help='Alpha')
	parser.add_argument('-t', '--train_propt', default=0.8)

	args = parser.parse_args()
	return args

def output_model_setting(args):
	print('Learning rate: {}'.format(args.learning_rate))
	print('Total number of epochs: {}'.format(args.epochs))
	print('Mini-batch size: {}'.format(args.batch_size))
	print('Train proportion: {}\n'.format(args.train_propt))
	print('Use cuda: {}'.format(use_cuda))

def train(model, optimizer, train_loader):
	#model.train()
	print('|\tTrain:')

	for batch_idx, (data, target) in enumerate(train_loader):

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		x, y1, y2, lam = mixup_data_and_target(data, target, alpha, use_cuda)
		pdb.set_trace()

if __name__ == '__main__':
	
	args = parse()

	global alpha
	alpha = args.alpha

	output_model_setting(args)

	train_loader, valid_loader, test_loader = load_cifar10_data(
														batch_size=args.batch_size,
														test_batch_size=args.batch_size,
														alpha=args.train_propt)

	train(None, None, train_loader)