import numpy as np
import torch
from torch import optim
import data_prep as dp
import models as m
import train
import utils as utils
import config
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
This module contains the user implementation of functions to be optimized by GPyOpt. Functions must accept
parameters passed via GPyOpt and return a single number which is being optimized. The function complexity 
depends largely on which hyperparameters are to be optimized.

The parameter template is stored in config.py. Once the functioned to be optimized is defined update the 
reference for `opt_func` in config.py

Example:
    f_opt_mnist: implements a CNN on MNIST data		

Note:
    GPyOpt passes the parameters variable as a nested np.ndarray, access with: parameters = parameters[0]
    Parameters are in the same order as defined in config.py

TODO:
    * Change data loading for f_opt_mnist in case no pre-processing needed
"""

@utils.call_counter
def f_opt_mnist(parameters):
	"""The below example fits a CNN of varying capacity and tunes some key hyperparameters"""
	parameters = parameters[0]  # np.ndarray passed in is nested
	print(f'---------Starting Bay opt call {f_opt_mnist.calls} with parameters: ---------')
	utils.print_params(parameters, config.opt_BO)
	# Data loading:
	# TODO: what if no pre-processing needed?
	train_ds, valid_ds = dp.download_mnist()
	train_dl, valid_dl = dp.get_data(train_ds, valid_ds, int(parameters[6]))
	train_dl, valid_dl = dp.WrappedDataLoader(train_dl, dp.preprocess), dp.WrappedDataLoader(valid_dl, dp.preprocess)
	# Model definition:
	model = m.Mnist_CNN(n_lay=int(parameters[2]),
						n_c=int(parameters[3]),
						n_fc=int(parameters[4]),
						dropout=parameters[5])
	model.to(dev)
	# Optimizer:
	opt = optim.SGD(model.parameters(), lr=parameters[0], momentum=parameters[1])
	scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
	# Fit:
	score, metric = train.fit(config.epochs, model, config.loss_func, scheduler, train_dl, valid_dl, train.accuracy)
	return np.array(score)