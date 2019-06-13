from GPyOpt.methods import BayesianOptimization
import GPyOpt
import data_prep as dp
import models as m
import train
import utils as utils
import config
import opt

def main():
	utils.setup_folders()
	if config.cleanup_models_dir: utils.clean_folder('models/')  # delete models from previous runs
	# GPyOpt function call:
	model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=True)
	optimizer = BayesianOptimization(f=config.opt_func,  # objective function
					 domain=config.opt_BO,
					 model_type='GP',
					 acquisition_type='EI',
					 normalize_Y=True,
					 acquisition_jitter=0.05,  # positive value to make acquisition more explorative
					 exact_feval=True,  # whether the outputs are exact
					 maximize=False,
					)
	optimizer.run_optimization(max_iter=config.max_iter)  # 5 initial exploratory points + max_iter
	# Post-run printing and plotting:
	if config.plot:
		optimizer.plot_acquisition()  # plots y normalized (i.e. deviates) only in 1d or 2d
		optimizer.plot_convergence()
	print('--------------------------------------------------')
	print('Optimal parameters from Bay opt are:')
	utils.print_params(optimizer.x_opt, config.opt_BO)
	return optimizer

if __name__ == '__main__':
	opt = main()