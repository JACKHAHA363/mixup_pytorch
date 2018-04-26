import numpy as np
import torch
from torch.autograd import Variable

def mixup_data_and_target(x, y, alpha, use_cuda):
	lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

	batch_size = x.size(0)
	permutation_list = Variable(torch.LongTensor(np.random.permutation(batch_size)), volatile=True)
	if use_cuda:
		permutation_list = permutation_list.cuda()

	x1, x2  = x, x.index_select(0, permutation_list)
	x = lam*x1+(1.0-lam)*x2
	y1, y2 = y, y.index_select(0, permutation_list)
	y = lam*y1+(1.0-lam)*y2
	return x, y1, y2, lam