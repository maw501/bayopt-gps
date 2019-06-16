import torch
import numpy as np

def loss_batch(model, loss_func, xb, yb, opt=None):
	"""Basic function computing the loss per mini-batch"""
	loss = loss_func(model(xb), yb)
	if opt is not None:
		loss.backward()
		opt.step()
		opt.zero_grad()
	return loss.item(), len(xb)

def accuracy(model, xb, yb):
	"""Accuracy metric"""
	preds = torch.argmax(model(xb), dim=1)
	return (preds == yb).float().mean()

def fit(epochs, model, loss_func, sched, train_dl, valid_dl, metric_func=None, save_folder=None):
	"""Function computing validation loss and a single metric to monitor"""
	for e in range(epochs):
		model.train()
		t_loss = 0
		for i, (xb, yb) in enumerate(train_dl):
			b_loss, _ = loss_batch(model, loss_func, xb, yb, sched.optimizer)
			t_loss += b_loss

		model.eval()
		with torch.no_grad():
			losses, nums = zip(
				*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
			)
			if metric_func is not None: metric = [metric_func(model, xb, yb) for xb, yb in valid_dl]
		val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # average mini-batches (may be different size)
		if metric_func is not None:
			val_metric = np.sum(np.multiply(metric, nums)) / np.sum(nums)
			print(f'e: {e+1}, trn loss: {(t_loss/((i+1))):.4f}, val loss: {val_loss:.4f}, val met: {val_metric:.4f}')
		else:
			print(f'e: {e+1}, trn loss: {(t_loss/((i+1))):.4f}, val loss: {val_loss:.4f}')
		sched.step()
	# Model saving - for inference only atm:
	if save_folder is not None: save(model, save_folder, f'{model.name}_val_loss_{val_loss:.4f}.pth')
	return val_loss

def save(model, save_folder, filename):
	"""Save model for inference only
	https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
	"""
	torch.save({'name': model.name,
				'model_init_args': model.init_args,
				'model_state_dict': model.state_dict()
				},
				save_folder+filename)
