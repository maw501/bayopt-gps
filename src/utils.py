import os

def print_params(p, opt_BO):
	bo_names = [d['name'] for d in opt_BO]
	print(f'{bo_names[0]}: {p[0]:.4f} | {bo_names[1]}: {p[1]:.4f} | {bo_names[2]}: {p[2]:.4f} | '
		  f'{bo_names[3]}: {p[3]:.4f} | {bo_names[4]}: {p[4]:.4f} | {bo_names[5]}: {p[5]:.4f}')

def call_counter(func):
	"""Call counter decorator"""
	def helper(*args, **kwargs):
		helper.calls += 1
		return func(*args, **kwargs)

	helper.calls = 0
	helper.__name__ = func.__name__
	return helper

def setup_folders():
	if not os.path.exists('models/'):
		os.mkdir('models/')
		print(f'Directory: models/ created')

def clean_folder(folder):
	print(f'Deleting models from: {folder}')
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e)