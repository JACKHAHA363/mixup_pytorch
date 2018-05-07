from src.util.load_data import load_cifar10_data
from src.util.util import mixup_data_and_target, mixup_loss
from src.model.resnet import ResNet18
from src.model.densenet import DensNet190

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
#from torchvision.models.resnet import resnet18
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os, copy
import numpy as np
import functools

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
loss_fn = F.cross_entropy

use_cuda = torch.cuda.is_available()
if not use_cuda:
	raise NotImplementedError
else:
	available_devices = []
	torch.cuda.empty_cache()
	available_devices  = list(range(torch.cuda.device_count()))

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
						help='Learning rate')
	parser.add_argument('-s', '--seed', default=123, type=int, help='Random seed')
	parser.add_argument('--batch_size', default=128, type=int,
						help='Mini-batch size')
	parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
	parser.add_argument('-a', '--alpha', default=0.8, type=float,
						help='Alpha')
	parser.add_argument('-t', '--train_propt', default=0.8, help='Train proportion')
	parser.add_argument('--print_freq', default=100, help='Print frequency')
	parser.add_argument('--data_set', default='cifar10', help='[cifar10, cifar100]')
	parser.add_argument('-r', '--result_path', default='./result/cifar10', type=str,
						help='Result path')
	parser.add_argument('--model_name', default='CIFAR10_compare', type=str,
						help='Model name')
	parser.add_argument('--model', default='resnet', type=str,
						help='[resnet | densenet]')

	args = parser.parse_args()
	return args

def output_model_setting(args):
	print('Learning rate: {}'.format(args.learning_rate))
	print('Total number of epochs: {}'.format(args.epochs))
	print('Mini-batch size: {}'.format(args.batch_size))
	print('Train proportion: {}\n'.format(args.train_propt))
	print('Use cuda: {}'.format(use_cuda))
	print('Number of GPUs: {}'.format(len(available_devices)))

def train_mixup(model, optimizer, train_loader):
	model.train()
	print('|\tTrain MIXUP:')

	total_loss, correct, count = 0, 0, 0

	for batch_idx, (data, target) in enumerate(train_loader):

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		x, y1, y2, lam = mixup_data_and_target(data, target, alpha, use_cuda)
		x, y1, y2 = map(functools.partial(Variable, requires_grad=False), (x, y1, y2))

		optimizer.zero_grad()
		pred_y = model(x)
		loss = mixup_loss(loss_fn, pred_y, y1, y2, lam)

		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		_, pred_c = torch.max(pred_y.data, 1)
		correct += (lam*(pred_c== y1.data).sum().item()+(1-lam)*(pred_c == y2.data).sum().item())
		count += data.size(0)

		if (batch_idx+1) % print_freq == 0:
			print('|\t\tMini-batch #{}: Loss={:.4f}\tAcc={:.4f}'.format(batch_idx+1,
																		total_loss/float(batch_idx+1),
																		correct/float(count)))

def train(model, optimizer, train_loader):
	model.train()

	total_loss, correct, count = 0, 0, 0

	for batch_idx, (data, target) in enumerate(train_loader):

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

		optimizer.zero_grad()
		pred_y = model(data)
		loss = loss_fn(pred_y, target)

		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		_, pred_c = torch.max(pred_y.data, 1)
		correct += (pred_c== target.data).sum().item()
		count += data.size(0)

		if (batch_idx+1) % print_freq == 0:
			print('|\t\tMini-batch #{}: Loss={:.4f}\tAcc={:.4f}'.format(batch_idx+1,
																		total_loss/float(batch_idx+1),
																		correct/float(count)))

def eval(model, data_loader):
	model.eval()

	total_loss, correct, count = 0, 0, 0

	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(train_loader):

			if use_cuda:
				data, target = data.cuda(), target.cuda()

			data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

			pred_y = model(data)
			loss = loss_fn(pred_y, target)

			total_loss += loss.item()
			_, pred_c = torch.max(pred_y.data, 1)
			correct += (pred_c== target.data).sum().item()
			count += data.size(0)
	return total_loss/float(len(train_loader)), correct/float(count)


if __name__ == '__main__':
	
	args = parse()

	if not os.path.exists(args.result_path):
   		os.makedirs(args.result_path)

	global alpha, print_freq
	alpha, print_freq = args.alpha, args.print_freq

	global result_path, model_name
	result_path, model_name = args.result_path, args.model_name

	output_model_setting(args)

	if args.data_set == 'cifar100':
		train_loader, valid_loader, test_loader = load_cifar100_data(
															batch_size=args.batch_size,
															test_batch_size=args.batch_size,
															alpha=args.train_propt)
	else:
		train_loader, valid_loader, test_loader = load_cifar10_data(
															batch_size=args.batch_size,
															test_batch_size=args.batch_size,
															alpha=args.train_propt)

	if args.model == 'resnet':
		model = ResNet18()
		model_orig = ResNet18()
	else:
		model = DensNet190()
		model_orig = DensNet190()

	if use_cuda:
		with torch.cuda.device(available_devices[-1]):
			model.cuda()
		with torch.cuda.device(available_devices[0]):
			model_orig.cuda()


	TRAIN_MIXUP = {'loss': [], 'accuracy': []}
	VALID_MIXUP = {'loss': [], 'accuracy': []}
	TRAIN_ORIG = {'loss': [], 'accuracy': []}
	VALID_ORIG = {'loss': [], 'accuracy': []}

	optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
	optimizer_orig = optim.Adam(params=model_orig.parameters(), lr=args.learning_rate)

	for epoch_i in range(0, args.epochs+1):

		print("| Epoch {}/{}:".format(epoch_i, args.epochs))
		print("|\tMIXUP")
		with torch.cuda.device(available_devices[-1]):
			if epoch_i > 0:
				train_mixup(model, optimizer, train_loader)

			train_loss, train_acc = eval(model, train_loader)
			valid_loss, valid_acc = eval(model, test_loader)
			print("|\t[Train] loss={:.4f}\tacc={:.4f}".format(train_loss, train_acc))
			print("|\t[Valid] loss={:.4f}\tacc={:.4f}".format(valid_loss, valid_acc))

			TRAIN_MIXUP['loss'].append(train_loss)
			TRAIN_MIXUP['accuracy'].append(train_acc)
			VALID_MIXUP['loss'].append(valid_loss)
			VALID_MIXUP['accuracy'].append(valid_acc)

		print("|\tORIGINAL")
		with torch.cuda.device(available_devices[0]):
			if epoch_i > 0:
				train(model_orig, optimizer_orig, train_loader)

			train_loss, train_acc = eval(model_orig, train_loader)
			valid_loss, valid_acc = eval(model_orig, test_loader)
			print("|\t[Train] loss={:.4f}\tacc={:.4f}".format(train_loss, train_acc))
			print("|\t[Valid] loss={:.4f}\tacc={:.4f}".format(valid_loss, valid_acc))

			TRAIN_ORIG['loss'].append(train_loss)
			TRAIN_ORIG['accuracy'].append(train_acc)
			VALID_ORIG['loss'].append(valid_loss)
			VALID_ORIG['accuracy'].append(valid_acc)

	plt.plot(list(range(0,args.epochs+1,1)), VALID_ORIG['loss'], 'ro-', label='original')
	plt.plot(list(range(0,args.epochs+1,1)), VALID_MIXUP['loss'], 'bs-', label='mixup')
	plt.title('average loss at each epoch')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(loc=1)
	plt.savefig(os.path.join(result_path, model_name+'_loss.png'))
	plt.clf()

	plt.plot(list(range(0,args.epochs+1,1)), VALID_ORIG['accuracy'], 'ro-', label='original')
	plt.plot(list(range(0,args.epochs+1,1)), VALID_MIXUP['accuracy'], 'bs-', label='mixup')
	plt.title('average classification accuracy at each epoch')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.legend(loc=4)
	plt.savefig(os.path.join(result_path, model_name+'_accuracy.png'))
	plt.clf()