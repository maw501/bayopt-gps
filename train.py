import torch
import numpy as np

def loss_batch(model, loss_func, xb, yb, opt=None):
	'''Basic function computing the loss per mini-batch'''
	loss = loss_func(model(xb), yb)
	if opt is not None:
		loss.backward()
		opt.step()
		opt.zero_grad()
	return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
	'''Basic fit function only calculating validation loss'''
	for epoch in range(epochs):
		model.train()
		for xb, yb in train_dl:
			loss_batch(model, loss_func, xb, yb, opt)

		model.eval()
		with torch.no_grad():
			losses, nums = zip(
				*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
			)
		val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # average mini-batches (may be different size)
		print(f'epoch: {epoch}, val loss {val_loss:.4f}')
	return val_loss

def accuracy(model, xb, yb):
	'''Accuracy metric'''
	preds = torch.argmax(model(xb), dim=1)
	return (preds == yb).float().mean()

def fit_metric(epochs, model, loss_func, opt, train_dl, valid_dl):
	'''Function computing validation loss and prediction accuracy'''
	for epoch in range(epochs):
		model.train()
		for xb, yb in train_dl:
			loss_batch(model, loss_func, xb, yb, opt)
			# TODO: print training loss

		model.eval()
		with torch.no_grad():
			losses, nums = zip(
				*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
			)
			acc = [accuracy(model, xb, yb) for xb, yb in valid_dl]
		val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # average mini-batches (may be different size)
		val_acc = np.sum(np.multiply(acc, nums)) / np.sum(nums)
		print(f'epoch: {epoch}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')
	return val_loss, val_acc