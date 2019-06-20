from GPyOpt.methods import BayesianOptimization
import GPyOpt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import torchvision
import numpy as np
import src.data_prep as dp
import src.models as m
import src.train as train
import src.utils as utils
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#************************************************************
# 					OPT FUNCTION                            #
#************************************************************
@utils.call_counter
def f_opt_mnist(parameters):
    """The below example fits a CNN of varying capacity and tunes some key hyperparameters

    This function contains the user implementation of functions to be optimized by GPyOpt. Functions must accept
    parameters passed via GPyOpt and return a single number which is being optimized. The function complexity
    depends largely on which hyperparameters are to be optimized.

    Once the functioned to be optimized is defined update the parameters in `opt_func` below

    Example:
        f_opt_mnist: implements a CNN on MNIST data

    Note:
        GPyOpt passes the parameters variable as a nested np.ndarray, access with: parameters = parameters[0]
        Parameters are in the same order as defined in config.py"""
    parameters = parameters[0]  # np.ndarray passed in is nested
    print(f'---------Starting Bay opt call {f_opt_mnist.calls} with parameters: ---------')
    utils.print_params(parameters, opt_BO)
    # Data loading:
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=dp.mnist_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(parameters[6]), shuffle=True, num_workers=8)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=dp.mnist_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=int(parameters[6]), shuffle=False, num_workers=8)
    train_dl, valid_dl = dp.WrapDL(trainloader, dp.to_gpu), dp.WrapDL(testloader, dp.to_gpu)

    # Model definition:
    model = m.FlexCNN(n_lay=int(parameters[2]),
                      n_c=int(parameters[3]),
                      n_fc=int(parameters[4]),
                      dropout=parameters[5]
                      )
    model.to(dev)
    # Optimizer:
    opt = optim.SGD(model.parameters(), lr=parameters[0], momentum=parameters[1])
    scheduler = optim.lr_scheduler.CyclicLR(opt, parameters[0], parameters[0]*3,
                                            step_size_up=int(len(train_dl)*int(parameters[6])*epochs*0.25))
    # Fit:
    score = train.fit(epochs, model, loss_func, scheduler, train_dl, valid_dl, train.accuracy, model_folder)
    return np.array(score)

#************************************************************
# 					CONFIGS                                 #
#************************************************************
"""
Parameter settings for a single run.

Once the functioned to be optimized is defined above update the reference for `opt_func` below.

Attributes:
    plot (bool): whether to show convergence plots after BO has run
    loss_func: NN loss function
    epochs: how many epochs to fit each optimization for
    max_iter: how many iterations of the BO to run - note 5 initial runs are conducted on top of max_iter
    opt_func: the function to be optimized, defined in opt.py
    opt_BO: list of dicts for GPyOpt
    	Example: https://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/GPyOpt_mixed_domain.ipynb

Note:
	Parameters are passed in the same order as defined in opt_BO

TODO:
    * Add ability to change BO parameters and kernels"""
plot = True
loss_func = F.cross_entropy
opt_func = f_opt_mnist
cleanup_models_dir = True
model_folder = 'models/mnist/'

opt_BO = [{'name': 'lr', 'type': 'continuous', 'domain': (0.05, 0.25)},
            {'name': 'mom', 'type': 'continuous', 'domain': (0.8, 0.92)},
            {'name': 'num_lay', 'type': 'discrete', 'domain': range(1, 4)},
            {'name': 'num_c', 'type': 'discrete', 'domain': range(8, 22, 2)},
            {'name': 'num_fc', 'type': 'discrete', 'domain': range(10, 105, 5)},
            {'name': 'dropout', 'type': 'discrete', 'domain': np.linspace(0, 0.4, 11)},
            {'name': 'bs', 'type': 'discrete', 'domain': range(64, 288, 32)},
          ]
#************************************************************
# 					RUN                                     #
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
                                     exact_feval=False,  # whether the outputs are exact
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