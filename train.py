import torch
import numpy as np

def loss_batch(model, loss_func, xb, yb, opt=None):
	"""Basic function computing the loss per mini-batch"""
	loss = loss_func(model(xb), yb)
	if opt is not None:
		loss.backward()
		opt.step()
		# import pdb; pdb.set_trace()
		opt.zero_grad()
	return loss.item(), len(xb)

def accuracy(model, xb, yb):
	"""Accuracy metric"""
	preds = torch.argmax(model(xb), dim=1)
	return (preds == yb).float().mean()

def fit(epochs, model, loss_func, sched, train_dl, valid_dl, metric_func=None):
	"""Function computing validation loss and a single metric to monitor"""
	for epoch in range(epochs):
		model.train()
		for xb, yb in train_dl:
			loss_batch(model, loss_func, xb, yb, sched.optimizer)
		# TODO: print training loss

		model.eval()
		with torch.no_grad():
			losses, nums = zip(
				*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
			)
			if metric_func is not None: metric = [metric_func(model, xb, yb) for xb, yb in valid_dl]
		val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # average mini-batches (may be different size)
		if metric_func is not None:
			val_metric = np.sum(np.multiply(metric, nums)) / np.sum(nums)
			print(f'epoch: {epoch}, val loss: {val_loss:.4f}, val acc: {val_metric:.4f}')
		else:
			print(f'epoch: {epoch}, val loss: {val_loss:.4f}')
		sched.step()
	# Model saving - for inference only atm:
	# torch.save(model.state_dict(), PATH)
	return val_loss, val_metric


# def get_model(model: nn.Module):
# 	"""Return the model maybe wrapped inside `model`."""
# 	return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model
#
# def is_pathlike(x:Any)->bool: return isinstance(x, (str, Path))
#
# def save(self, file: PathLikeOrBinaryStream = None, return_path: bool = False, with_opt: bool = True):
# 	"""Save model and optimizer state (if `with_opt`) with `file` to `self.model_dir`. `file` can be file-like
# 	(file or buffer)"""
# 	if is_pathlike(file): self._test_writeable_path()
# 	if rank_distrib(): return  # don't save if slave proc
# 	target = self.path / self.model_dir / f'{file}.pth' if is_pathlike(file) else file
# 	if not hasattr(self, 'opt'): with_opt = False
# 	if not with_opt:
# 		state = get_model(self.model).state_dict()
# 	else:
# 		state = {'model': get_model(self.model).state_dict(), 'opt': self.opt.state_dict()}
# 	torch.save(state, target)
# 	if return_path: return target


# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             ...
#             }, PATH)