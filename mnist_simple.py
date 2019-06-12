import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from GPyOpt.methods import BayesianOptimization
import data_prep as dp
import models as m
import train
import utils as utils

@utils.call_counter
def f_opt(parameters):
	'''
	******************* FUNCTION TO CREATE *******************
	Function that takes in a list of parameters as specified by opt_BO and outputs a single number to optimize,
	 	complexity depends on which hyperparameters are to be optimized

	The below example fits a CNN of varying capacity and tunes some key hyperparameters
	'''
	parameters = parameters[0]  # np.ndarray passed in is nested
	print(f'---------Starting Bay opt call {f_opt.calls} with parameters: ---------')
	utils.print_params(parameters, opt_BO)
	train_dl, valid_dl = dp.get_data(train_ds, valid_ds, int(parameters[6]))
	train_dl, valid_dl = dp.WrappedDataLoader(train_dl, dp.preprocess), dp.WrappedDataLoader(valid_dl, dp.preprocess)
	model = m.Mnist_CNN(n_lay=int(parameters[2]),
						n_c=int(parameters[3]),
						n_fc=int(parameters[4]),
						dropout=parameters[5])
	model.to(dev)
	opt = optim.SGD(model.parameters(), lr=parameters[0], momentum=parameters[1])
	scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
	score, metric = train.fit_metric(epochs, model, loss_func, opt, train_dl, valid_dl)
	return np.array(score)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_ds, valid_ds = dp.download_mnist()

# CONFIG
interactive = False
plot = True
loss_func = F.nll_loss

opt_BO = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0.05, 0.25)},
	  {'name': 'momentum', 'type': 'continuous', 'domain': (0.8, 0.9)},
	  {'name': 'num_lay', 'type': 'discrete', 'domain': range(1, 4)},
	  {'name': 'num_c', 'type': 'discrete', 'domain': range(8, 22, 2)},
	  {'name': 'num_fc', 'type': 'discrete', 'domain': range(10, 105, 5)},
	  {'name': 'dropout', 'type': 'discrete', 'domain': np.linspace(0, 0.4, 11)},
	  {'name': 'bs', 'type': 'discrete', 'domain': range(64, 288, 32)}
	  ]

def main():
	global epochs
	if not interactive:
		epochs, max_iter = int(sys.argv[1]), int(sys.argv[2])
	else:
		epochs, max_iter = 5, 5
	optimizer = BayesianOptimization(f=f_opt,  # objective function
					 domain=opt_BO,
					 model_type='GP',
					 acquisition_type='EI',
					 normalize_Y=True,
					 acquisition_jitter=0.05,  # positive value to make acquisition more explorative
					 exact_feval=True,  # whether the outputs are exact
					 maximize=False
					)

	optimizer.run_optimization(max_iter=max_iter)  # 5 initial exploratory points + max_iter
	if plot:
		optimizer.plot_acquisition()  # plots y normalized (i.e. deviates) only in 1d or 2d
		optimizer.plot_convergence()
	print('--------------------------------------------------')
	print('Optimal parameters from Bay opt are:')
	utils.print_params(optimizer.x_opt, opt_BO)
	return optimizer

if __name__ == '__main__':
	opt = main()