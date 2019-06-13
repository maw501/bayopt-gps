import torch.nn.functional as F
import numpy as np
import opt
#    ______   ______   .__   __.  _______  __    _______      _______.
#   /      | /  __  \  |  \ |  | |   ____||  |  /  _____|    /       |
#  |  ,----'|  |  |  | |   \|  | |  |__   |  | |  |  __     |   (----`
#  |  |     |  |  |  | |  . `  | |   __|  |  | |  | |_ |     \   \
#  |  `----.|  `--'  | |  |\   | |  |     |  | |  |__| | .----)   |
#   \______| \______/  |__| \__| |__|     |__|  \______| |_______/
"""
Parameter settings for a single run.

The parameter template is stored in config.py. Once the functioned to be optimized is defined update the 
reference for `opt_func` in config.py

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
    * Add ability to change BO parameters and kernels
"""
plot = True
loss_func = F.nll_loss
epochs = 2
max_iter = 2
opt_func = opt.f_opt_mnist

opt_BO = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0.05, 0.25)},
		  {'name': 'momentum', 'type': 'continuous', 'domain': (0.8, 0.9)},
		  {'name': 'num_lay', 'type': 'discrete', 'domain': range(1, 4)},
		  {'name': 'num_c', 'type': 'discrete', 'domain': range(8, 22, 2)},
		  {'name': 'num_fc', 'type': 'discrete', 'domain': range(10, 105, 5)},
		  {'name': 'dropout', 'type': 'discrete', 'domain': np.linspace(0, 0.4, 11)},
		  {'name': 'bs', 'type': 'discrete', 'domain': range(64, 288, 32)}
		  ]