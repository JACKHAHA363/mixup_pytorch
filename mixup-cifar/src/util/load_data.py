import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random

import numpy as np
import pdb

def load_cifar10_data(batch_size, test_batch_size, alpha=1):

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])


	train_data = dset.CIFAR10(root='data', train=True, download=True, transform=transform_train)
	train_len = int(len(train_data)*alpha)
	train_data.train_data = train_data.train_data[:train_len]
	train_data.train_labels = train_data.train_labels[:train_len]
	valid_data = dset.CIFAR10(root='data', train=True, download=True, transform=transform_train)
	valid_data.train_data = valid_data.train_data[train_len:]
	valid_data.train_labels = valid_data.train_labels[train_len:]

	test_data = dset.CIFAR10(root='data', train=False, download=True, transform=transform_test)

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, valid_loader, test_loader

def load_cifar100_data(batch_size, test_batch_size, alpha=1):

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])


	train_data = dset.CIFAR100(root='data', train=True, download=True, transform=transform_train)
	train_len = int(len(train_data)*alpha)
	train_data.train_data = train_data.train_data[:train_len]
	train_data.train_labels = train_data.train_labels[:train_len]
	valid_data = dset.CIFAR100(root='data', train=True, download=True, transform=transform_train)
	valid_data.train_data = valid_data.train_data[train_len:]
	valid_data.train_labels = valid_data.train_labels[train_len:]

	test_data = dset.CIFAR100(root='data', train=False, download=True, transform=transform_test)

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, valid_loader, test_loader


def load_cifar10_by_class(batch_size, test_batch_size):
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])


	train_data = dset.CIFAR10(root='data', train=True, download=True, transform=transform_train)

	sort_idx = np.argsort(train_data.train_labels)
	train_data.train_data = train_data.train_data[sort_idx]
	train_data.train_labels = list(np.array(train_data.train_labels)[sort_idx])

	change_idx = []
	curr_label = 0

	for idx, label in enumerate(train_data.train_labels):
		if label != curr_label:
			change_idx.append(idx)
			curr_label = label

	change_idx.append(len(train_data.train_labels))

	pair_idx = []
	pointer_idx = 0
	prev_pointer = 0
	last_pointer = change_idx[pointer_idx]

	for i in range(len(train_data.train_labels)):

		if i >= last_pointer:
			pointer_idx += 1
			prev_pointer = last_pointer
			last_pointer = change_idx[pointer_idx]

		pair_idx.append(random.randint(prev_pointer, last_pointer-1))

	new_data_idx = list(range(len(train_data.train_labels)))
	random.shuffle(new_data_idx)

	train_data2 = dset.CIFAR10(root='data', train=True, download=False, transform=transform_train)
	train_data2.train_data = train_data2.train_data[sort_idx][pair_idx][new_data_idx]
	train_data2.train_labels = list(np.array(train_data2.train_labels)[sort_idx][pair_idx][new_data_idx])

	train_data.train_data = train_data.train_data[new_data_idx]
	train_data.train_labels = list(np.array(train_data.train_labels)[new_data_idx])

	test_data = dset.CIFAR10(root='data', train=False, download=True, transform=transform_test)

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=False, drop_last=True)
	train_loader2 = torch.utils.data.DataLoader(
		train_data2, batch_size=batch_size, shuffle=False, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, train_loader2, test_loader

