from GPyOpt.methods import BayesianOptimization
import GPyOpt
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import src.data_prep as dp
import src.models as m
import src.train as train
import src.utils as utils
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#************************************************************
# 					OPT FUNCTION							#
#************************************************************
@utils.call_counter
def f_opt_cifar(parameters):
	parameters = parameters[0]  # np.ndarray passed in is nested
	print(f'---------Starting Bay opt call {f_opt_cifar.calls} with parameters: ---------')
	utils.print_params(parameters, opt_BO)
	# Data loading:
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=dp.pytorch_transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(parameters[6]), shuffle=True, num_workers=8)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=dp.pytorch_transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=int(parameters[6]), shuffle=False, num_workers=8)
	train_dl, valid_dl = dp.WrapDL(trainloader, dp.to_gpu), dp.WrapDL(testloader, dp.to_gpu)
	# Model definition:
	model = m.FlexCNN(n_lay=int(parameters[2]),
						n_c=int(parameters[3]),
						n_fc=int(parameters[4]),
						dropout=parameters[5],
					 	chw=(3,32,32))
	model.to(dev)
	# Optimizer:
	opt = optim.SGD(model.parameters(), lr=parameters[0], momentum=parameters[1])
	scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=parameters[7])
	# Fit:
	score = train.fit(epochs, model, loss_func, scheduler, train_dl, valid_dl, train.accuracy, model_folder)
	return np.array(score)
#************************************************************
# 					CONFIGS									#
#************************************************************
plot = True
loss_func = F.cross_entropy
opt_func = f_opt_cifar
cleanup_models_dir = True
model_folder = 'models/cifar10/'
download = False  # set to True for first run

opt_BO = [{'name': 'lr', 'type': 'continuous', 'domain': (0.001, 0.1)},
		  {'name': 'mom', 'type': 'continuous', 'domain': (0.9, 1)},
		  {'name': 'num_lay', 'type': 'discrete', 'domain': range(2, 6)},
		  {'name': 'num_c', 'type': 'discrete', 'domain': range(8, 72, 8)},
		  {'name': 'num_fc', 'type': 'discrete', 'domain': range(128, 384, 128)},
		  {'name': 'dropout', 'type': 'discrete', 'domain': np.linspace(0, 0.5, 11)},
		  {'name': 'bs', 'type': 'discrete', 'domain': range(256, 768, 256)},
		  {'name': 'lr_decay', 'type': 'continuous', 'domain': (0.9, 1)}
		  ]
#************************************************************
# 					RUN										#
#************************************************************
def main():
	utils.setup_folders(model_folder)
	if cleanup_models_dir: utils.clean_folder(model_folder)  # delete models from previous runs
	# GPyOpt function call:
	optimizer = BayesianOptimization(f=opt_func,  # objective function
					 domain=opt_BO,
					 model_type='GP',
					 acquisition_type='EI',
					 normalize_Y=True,
					 acquisition_jitter=0.05,  # positive value to make acquisition more explorative
					 exact_feval=True,  # whether the outputs are exact
					 maximize=False,
					)
	optimizer.run_optimization(max_iter=max_iter)  # 5 initial exploratory points + max_iter
	# Post-run printing and plotting:
	if plot:
		optimizer.plot_acquisition()  # plots y normalized (i.e. deviates) only in 1d or 2d
		optimizer.plot_convergence()
	print('--------------------------------------------------')
	print('Optimal parameters from Bay opt are:')
	utils.print_params(optimizer.x_opt, opt_BO)
	return optimizer

if __name__ == '__main__':
	args = utils.parser.parse_args()
	epochs, max_iter = args.epochs, args.max_iter
	opt = main()