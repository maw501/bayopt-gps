from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class ConvBnRelu(nn.Module):
	"""Re-usable building block implementing 2d conv layer with batch-norm and relu activation"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=5),  # reduces feature map by 4
								  nn.BatchNorm2d(out_channels),
								  nn.ReLU(inplace=True)
								  )

	def forward(self, x):
		return self.conv(x)

class Mnist_CNN(nn.Module):
	"""Simple CNN for MNIST data of size 28x28"""
	def __init__(self, n_lay=2, n_c=16, n_fc=25, dropout=0.1):
		super().__init__()
		self.name = 'mnist_cnn'
		self.init_args = {'n_lay':n_lay, 'n_c':n_c, 'n_fc':n_fc}  # for loading back model with correct size
		self.h, self.w = 28, 28
		self.out_hw = self.h - (n_lay + 1) * 4
		self.out_c = n_c * (2 ** n_lay)
		self.convs = nn.ModuleList([ConvBnRelu(1, n_c)])
		self.convs.extend([ConvBnRelu((2 ** (i - 1)) * n_c, (2 ** i) * n_c) for i in range(1, n_lay + 1)])
		self.fc1 = nn.Linear(self.out_c * (self.out_hw ** 2), n_fc * 2)
		self.fc2 = nn.Linear(n_fc * 2, 10)

	def forward(self, x):
		bs = x.shape[0]
		x = x.view(-1, 1, self.h, self.w)
		for i, l in enumerate(self.convs):
			x = self.convs[i](x)
		x = x.view(bs, -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class Net(nn.Module):
	"""NN from PyTorch MNIST tutorial"""
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4 * 4 * 50, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4 * 4 * 50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)