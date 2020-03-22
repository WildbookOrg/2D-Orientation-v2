#!/usr/bin/env python3
# ok

"""

example:
python3 train.py --type regression --nClasses 2 --device 0 --separate-trig --batchSz 3 --animal seadragon --example

training:
srun --time=240 --gres=gpu:1 --ntasks=1 python3 train.py --type regression --nClasses 2 --device 0 --separate-trig --batchSz 50 --animal seadragon

loss history
python3 train.py --type regression --nClasses 2 --separate-trig --pretrain --plot-loss-history --animal seadragon

"""

import argparse
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import os
import sys
import math
import shutil

from utils.test import test_stats

# imort data loader for both mnist and turtles
from data import *

import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')

"""
TODO

add tensorboard tracking and retrain everything
better augmentation 
solidify the separate trig and degree loss hyperparams
separate everything into classes then functions then steps
"""

def arg_check(args):
	# no-resume and save_path exists, check if they meant it
	if(args.no_resume and os.path.exists(args.save_path)):
		# i = input("Warning: Deleting Save Directory. Continue? (y/n) ")
		
		# accident, dont delete save directory
		# if(i!='y'):
		# 	print("Program Cancelled")
		# 	exit(1)	

		# delete save directory
		# else:
		shutil.rmtree(args.save_path)

	# make an empty save path if not resuming
	if(args.no_resume):
		os.makedirs(args.save_path, exist_ok=True)

	# resuming but no pth file found
	if(not args.no_resume and not os.path.exists(args.pth_file)):
		print('No pth file found, use --no-resume')
		exit(1)

	# if(args.pretrain and not args.no_resume):
	# 	print('When using --pretrain, use --no-resume as well')
	# 	exit(1)

	if(args.type.startswith('regression') and args.type.endswith('degree')):
		args.degree_loss = True

	if(args.nClasses == None):
		if(args.type.startswith('classification')):
			if(args.type.endswith('8')):
				args.nClasses = 8
			elif(args.type.endswith('4')):
				args.nClasses = 4
			else:
				args.nClasses = 360
		elif(args.type.startswith('regression')):
			args.nClasses = 1
		else:
			print("Invalid class number provided, see -h for help")
			exit(1)

	if(args.separate_trig):
		args.nClasses=2

def example_function(model, args):
	
	test_dataset = Data_turtles(dataType='test2020',experiment_type='example', args = args)
	if(args.batchSz>8):
		args.batchSz=8
	testLoader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=True)
	test_iter = iter(testLoader)

	model.eval()
	images_normalized, images, angles= next(test_iter)

	if args.cuda:
			images_normalized, angles = images_normalized.cuda(), angles.cuda()
	data, target = Variable(images_normalized), Variable(angles)
	
	data = data.to(args.device)
	target = target.to(args.device)

	output = model(data)
	output = output.cpu()
	print(output)
	print(target)

	if(args.type.startswith('classification')):
		pred = output.data.max(1)[1] # get the index of the max log-probability
		print(pred*(360//args.nClasses))
	if(args.type.startswith('regression')):
		if(args.separate_trig):
			pred = separate_trig_to_angle(output, args)
		else:
			pred = output.data.reshape((1,args.batchSz))[0]


	print(pred)

	images = [transforms.ToPILImage()(image) for image in images]	
	images = [transforms.functional.affine(image,-angle,(0,0),1,0) for image,angle in zip(images,pred)]
	images = [transforms.ToTensor()(image) for image in images]	

	grid = utils.make_grid(images)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	plt.axis('off')
	plt.title(args.type+'\nPred: {}\nTrue: {}'.format(angles.detach().cpu().numpy(),pred.detach().cpu().int().numpy()))
	plt.show()  

def save_all_figures(model, args):
	test_dataset = Data_turtles(dataType='test2020',experiment_type='example', args = args)
	testLoader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=False)

	model.eval()
	i_major = 0
	for images_normalized, images, angles in testLoader:


		if args.cuda:
				images_normalized, angles = images_normalized.cuda(), angles.cuda()
	
		data = images_normalized.to(args.device)
		target = angles.to(args.device)

		output = model(data)
		output = output.cpu()
	

		if(args.type.startswith('classification')):
			pred = output.data.max(1)[1] # get the index of the max log-probability
			print(pred*(360//args.nClasses))
		if(args.type.startswith('regression')):
			if(args.separate_trig):
				pred = separate_trig_to_angle(output, args)
			else:
				pred = output.data.reshape((1,args.batchSz))[0]



		images = [transforms.ToPILImage()(image) for image in images]	
		images = [transforms.functional.affine(image,-angle,(0,0),1,0) for image,angle in zip(images,pred)]
		images = [transforms.ToTensor()(image) for image in images]	


		for i in range(args.batchSz):
			if(angles[i] - pred[i] >7):
				plt.imshow(images[i].numpy().transpose((1, 2, 0)))
				plt.axis('off')
				plt.title('Regression Trig Loss'+'\nTrue: '+str(int(angles[i]))+'       \nPred: {:.3f}'.format(float(pred[i])))
				plt.savefig('./results/turtle/reg_trig_figures/bad_example{}.png'.format(str(i_major+i)))  

		i_major+=args.batchSz
		print(i_major,'/',len(testLoader.dataset))


def get_data_loaders(args):
	train_dataset = Data_turtles(dataType = 'train2020', experiment_type='train', args = args)
	val_dataset = Data_turtles(dataType = 'val2020', experiment_type='validation', args = args)
	test_dataset = Data_turtles(dataType='test2020', experiment_type='test', args = args)

	dataloaders = {
		'train' : DataLoader(train_dataset,batch_size=args.batchSz,shuffle=True),
		'val' : DataLoader(val_dataset,batch_size=args.batchSz,shuffle=True)
	}		
	return dataloaders

def get_text_files(args):
	datafiles = {
		'train' : open(os.path.join(args.save_path, 'train.csv'), 'a'),
		'val' : open(os.path.join(args.save_path, 'val.csv'), 'a')
	}
	return datafiles

def get_optimizer(model, args):
	if args.opt == 'sgd':
		return optim.SGD(model.parameters(), lr=1e-1,
				momentum=0.9, weight_decay=1e-4)
	elif args.opt == 'adam':
		return optim.Adam(model.parameters(), weight_decay=1e-4)
	elif args.opt == 'rmsprop':
		return optim.RMSprop(model.parameters(), weight_decay=1e-4)
	
def plot_loss_history(args):
	datafiles = {
		'train' : open(os.path.join(args.save_path, 'train.csv'), 'r'),
		'val' : open(os.path.join(args.save_path, 'val.csv'), 'r')
	}

	# partial epoch, loss, error

	train_partial_epoch = []
	train_partial_loss = []
	train_epoch = []
	train_loss = []

	val_partial_epoch = []
	val_partial_loss = []
	val_epoch = []
	val_loss = []

	train_file = datafiles['train']
	for line in train_file:
		line = [float(x) for x in line.strip().split(',')]
		# print(line)
		if(len(line)==3):
			partialEpoch, loss, error = line
			train_partial_epoch.append(partialEpoch)
			train_partial_loss.append(loss)

		if(len(line)==2):
			epoch, loss = line
			train_epoch.append(epoch)
			train_loss.append(loss)

	val_file = datafiles['val']
	for line in val_file:
		line = [float(x) for x in line.strip().split(',')]
		# print(line)
		if(len(line)==3):
			partialEpoch, loss, error = line
			val_partial_epoch.append(partialEpoch)
			val_partial_loss.append(loss)

		if(len(line)==2):
			epoch, loss = line
			val_epoch.append(epoch)
			val_loss.append(loss)

	train_partial_epoch_split = []
	train_partial_loss_split = []
	prev_index = 0
	for i in range(1,len(train_partial_epoch)):
		if(train_partial_epoch[i]<train_partial_epoch[i-1]):
			train_partial_epoch_split.append(train_partial_epoch[prev_index:i])
			train_partial_loss_split.append(train_partial_loss[prev_index:i])
			prev_index = i
	if(len(train_partial_epoch_split)==0):
		train_partial_epoch_split.append(train_partial_epoch)
		train_partial_loss_split.append(train_partial_loss)

	val_partial_epoch_split = []
	val_partial_loss_split = []
	prev_index = 0
	for i in range(1,len(val_partial_epoch)):
		if(val_partial_epoch[i]<val_partial_epoch[i-1]):
			val_partial_epoch_split.append(val_partial_epoch[prev_index:i])
			val_partial_loss_split.append(val_partial_loss[prev_index:i])
			prev_index = i
	if(len(val_partial_epoch_split)==0):
		val_partial_epoch_split.append(val_partial_epoch)
		val_partial_loss_split.append(val_partial_loss)

	for i in range(len(train_partial_epoch_split)):
		plt.plot(train_partial_epoch_split[i], train_partial_loss_split[i],label='train partial',color='b')
	for i in range(len(val_partial_epoch_split)):
		plt.plot(val_partial_epoch_split[i], val_partial_loss_split[i],label='val partial',color='r')
	
	plt.title('History of Each Batch')
	plt.xlabel('Partial Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

	train_epoch_split = []
	train_loss_split = []
	prev_index = 0
	for i in range(1,len(train_epoch)):
		if(train_epoch[i]<train_epoch[i-1]):
			train_epoch_split.append(train_epoch[prev_index:i])
			train_loss_split.append(train_loss[prev_index:i])
			prev_index = i
	if(len(train_epoch_split)==0):
		train_epoch_split.append(train_epoch)
		train_loss_split.append(train_loss)

	val_epoch_split = []
	val_loss_split = []
	prev_index = 0
	for i in range(1,len(val_epoch)):
		if(val_epoch[i]<val_epoch[i-1]):
			val_epoch_split.append(val_epoch[prev_index:i])
			val_loss_split.append(val_loss[prev_index:i])
			prev_index = i
	if(len(val_epoch_split)==0):
		val_epoch_split.append(val_epoch)
		val_loss_split.append(val_loss)

	for i in range(len(train_epoch_split)):
		plt.plot(train_epoch_split[i], train_loss_split[i],label='train',color='b')
	for i in range(len(val_epoch_split)):
		plt.plot(val_epoch_split[i], val_loss_split[i],label='val',color='r')
	
	saved_epoch = []
	saved_loss = []
	best = val_loss[0]
	for i in range(1,len(val_epoch)):
		if(val_loss[i]<best):
			saved_epoch.append(val_epoch[i])
			saved_loss.append(val_loss[i])
			best = val_loss[i]

	saved_epoch_split = []
	saved_loss_split = []
	prev_index = 0
	for i in range(1,len(saved_epoch)):
		if(saved_epoch[i]<saved_epoch[i-1]):
			saved_epoch_split.append(saved_epoch[prev_index:i])
			saved_loss_split.append(saved_loss[prev_index:i])
			prev_index = i
	if(len(saved_epoch_split)==0):
		saved_epoch_split.append(saved_epoch)
		saved_loss_split.append(saved_loss)

	for i in range(len(saved_epoch_split)):
		plt.plot(saved_epoch_split[i],saved_loss_split[i],label='saved',color='g')


	plt.title('History of Each Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batchSz', type=int, default=60, help='specify a batch size')
	parser.add_argument('--nEpochs', type=int, default=25, help='specify a number of epochs')
	parser.add_argument('--opt', type=str, default='adam', 
							choices=('sgd', 'adam', 'rmsprop'))
	parser.add_argument('--no-resume', action='store_true', help='delete current trained weights and train from scratch')
	parser.add_argument('--example', action='store_true', help='show a visualization of a sample of the test set')
	parser.add_argument('--type', default='regression', help='specify a network type to be used with directory saving')
	parser.add_argument('--test', action='store_true', help='test the current weights and plot some statistics')
	parser.add_argument('--nClasses', type=int, default=1, help='specify the number of classes (must be 1 for regression)')
	parser.add_argument('--pretrain', action='store_true', help='used with --no-resume, load wiped network with densenet pretrained values')
	parser.add_argument('--separate-trig', action='store_true', help='use the trig components (sin,cos) to estimate rotation, rather than theta')
	parser.add_argument('--degree-loss', action='store_true',help='estimate orientation using sin cos components')
	parser.add_argument('--device', type=int, default=0, choices=tuple(range(torch.cuda.device_count())), help='specify the cuda device to run on')
	parser.add_argument('--save-all-figs', action='store_true', help='apply the example function to each test image and save in a directory') # TODO 
	parser.add_argument('--animal', type=str, default='seaturtle', 
							choices=('seaturtle', 'turtles1','seadragon','mantaray','rightwhale'))
	parser.add_argument('--show', action='store_true',help='shows each image during dataloader stage to see annotation')
	parser.add_argument('--plot-loss-history', action='store_true')
	
	parser.add_argument('--lr', type=float, default=1e-4, help='specify the initial learning rate')
	parser.add_argument('--angle-range', type=int, default=50, help='angle range to train with; within [2,360]')

	args = parser.parse_args()


	# save a bunch of constants to avoid confusion later
	args.cuda = torch.cuda.is_available()
	args.save_path = 'work/densenet.%s.%s.%s%s/'%(args.animal,args.type,args.nClasses,'.pretrain' if args.pretrain else '')
	args.save_file = '{}.{}.latest.pth'.format(args.type, args.nClasses)
	args.pth_file = os.path.join(args.save_path,args.save_file)
	arg_check(args)

	if(args.plot_loss_history):
		plot_loss_history(args)
		exit(1)

	

	# create model instance, add ending classifier
	try:
		model = torchvision.models.densenet161(pretrained=args.pretrain)
	except:
		print('type this: \n\texport TORCH_HOME=~/.torch')
		exit(1)
	model.classifier = nn.Linear(2208, args.nClasses)
	if(args.type.startswith('classification')):
		model.classifier = nn.Sequential(
			nn.Linear(2208, args.nClasses), 
			nn.LogSoftmax(dim=0)
		)

	# resume training by loading state dict
	if not args.no_resume:
		model.load_state_dict(torch.load(args.pth_file))
		
	model = model.to(args.device)
	
	if(args.save_all_figs):
		save_all_figures(model, args)
		exit(1)

	if args.example:
		example_function(model, args)
		exit(1)

	# load files to track loss progress
	datafiles = get_text_files(args)

	if(args.test):
		test(args, 1, model)
		exit(1)

	# load dataloaders that hold each data set
	dataloaders = get_data_loaders(args)


	if args.cuda:
		model = model.cuda()
	optimizer = get_optimizer(model, args)
	
	if(args.type.startswith('classification')):
		loss_func = F.nll_loss
	elif(args.separate_trig):
		loss_func = trig_loss
	else:
		loss_func = mse
	
	# training loop
	best_val_loss = train(args, 0, model, dataloaders['val'], optimizer, datafiles['val'], loss_func, 'val')
	print('best val loss:',best_val_loss)
	print()
	for epoch in range(1, args.nEpochs + 1):
		adjust_opt(args.opt, optimizer, epoch)
		for phase in ['train','val']:
			val_loss = train(args, epoch, model, dataloaders[phase], optimizer, datafiles[phase], loss_func, phase)
			
		if(val_loss<best_val_loss):
			print("Saving State Dict: new loss is",val_loss)
			best_val_loss = val_loss
			torch.save(model.state_dict(), args.pth_file)
		else:
			print('Loss was:', val_loss)
			print()
		
	for key in datafiles.keys():
		datafiles[key].close()

def trig_loss(t1, t2, args):
	assert(t1.size(1)==2)
	target = angle_to_separate_trig(t2, args)
	return mse(t1,target, args)

def angle_to_separate_trig(t1, args):
	t1 = t1.reshape((t1.size(0),1))
	t1 = deg_to_radian(t1)
	t1 = torch.cat((torch.cos(t1),torch.sin(t1)),dim=1)
	if(args.degree_loss):
		t1 = radian_to_deg(t1)
	return t1

# use atan2 TODO

def separate_trig_to_angle(t1, args):
	assert(t1.size(1)==2)
	if(args.degree_loss):
		t1 = deg_to_radian(t1)
	# print('t1:',t1)
	# t1[:,0] = torch.acos(t1[:,0])
	# t1[:,1] = torch.asin(t1[:,1])
	# print('t1::',t1)
	# t1 = radian_to_deg(t1)
	# return torch.where(t1[:,1]>0, t1[:,0], 360 - t1[:,0])
	x = radian_to_deg(torch.atan2(t1[:,1],t1[:,0])) 
	return torch.where(x<0, 360+x, x)


def deg_to_radian(deg):
	return (deg*np.pi)/180

def radian_to_deg(rad):
	return (rad*180)/np.pi

def mse(t1, t2, args=None):
	diff = torch.abs(t1.squeeze()-t2.squeeze())
	torch.where(diff>180,360-diff,diff)
	m = torch.mean(diff)
	return m*m

def L1(t1, t2):
	diff = torch.abs(t1.squeeze()-t2.squeeze())
	torch.where(diff>180,360-diff,diff)
	return torch.mean(diff)

def difference(args,t1,t2):
	diff = abs(t1-t2)
	# if(args.type.endswith("45")):
	# 	diff = np.where(diff>90,180-diff,diff)
	# else:
	diff = np.where(diff>180,360-diff,diff)
	return diff

def train(args, epoch, net, dataloader, optimizer, datafile, loss_func, phase):
	if(phase == 'train'):
		net.train()
	else:
		net.eval()

	# net = net.to(args.device)

	val_loss = 0
	nProcessed = 0
	incorrect = 0
	nTrain = len(dataloader.dataset)
	for batch_idx, (data, target) in enumerate(dataloader):
		nProcessed += len(data)
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		# data, target = Variable(data), Variable(target)
		data = data.to(args.device)
		target = target.to(args.device)
		optimizer.zero_grad()
		output = net(data)

		if(args.type.startswith('classification')):
			loss = loss_func(output, target)
			pred = output.data.max(1)[1] # get the index of the max log-probability
			incorrect += pred.ne(target.data).cpu().sum()
			err = torch.mean(abs(pred.float() - target.float()))
			val_loss += loss
		if(args.type.startswith('regression')):
			loss = loss_func(output.float(), target.float(), args)
			val_loss += loss
			pred = output.data.squeeze()

			if(args.separate_trig):
				err = torch.mean(abs(output.float() - angle_to_separate_trig(target.float(), args)))
			else:
				err = torch.mean(abs(output.float().squeeze() - target.float()))
			
			
		if(phase == 'train'):
			loss.backward()
			optimizer.step()

	
			
		partialEpoch = epoch + batch_idx / len(dataloader) - 1
		print(phase+': {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
			partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(dataloader),
			loss.data, err.item()))


		datafile.write('{},{},{}\n'.format(partialEpoch, loss.data, err))
		datafile.flush()	

	datafile.write('{},{}\n'.format(epoch,val_loss))
	print(epoch,val_loss)

	return val_loss

# def val(args, epoch, net, valLoader, optimizer, valF, loss_func):
# 	val_loss = 0
# 	nProcessed = 0
# 	incorrect = 0
# 	nTrain = len(valLoader.dataset)
# 	for batch_idx, (data, target) in enumerate(valLoader):
# 		if args.cuda:
# 			data, target = data.cuda(), target.cuda()
# 		data, target = Variable(data), Variable(target)
# 		output = net(data)
# 		if(args.type.startswith('classification')):
# 			val_loss += loss_func(output, target).data
# 			pred = output.data.max(1)[1] # get the index of the max log-probability
# 			incorrect += pred.ne(target.data).cpu().sum()
# 		if(args.type.startswith('regression')):
# 			val_loss += loss_func(output, target.float()).data
# 			pred = output.data.squeeze()

# 	val_loss /= len(valLoader)
# 	return val_loss

def test(args, epoch, net):
	test_dataset = Data_turtles(dataType='test2020', experiment_type='test', args = args)
	dataloader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=False, drop_last=True)
	testF = open(os.path.join(args.save_path, 'test.csv'), 'w')
	net.eval()
	test_loss = 0
	all_pred = None
	all_targ = None
	all_diff = None
	i_major = 0
	for data, display_image, target in dataloader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		# data, target = Variable(data), Variable(target)
		data = data.to(args.device)
		target = target.to(args.device)
		output = net(data)

		if(args.type.startswith('classification')):
			test_loss += F.nll_loss(output, target).data
			pred = output.data.max(1)[1] # get the index of the max log-probability
			incorrect += pred.ne(target.data).cpu().sum()
		if(args.type.startswith('regression')):
			if(args.separate_trig):
				test_loss += trig_loss(output.float(), target.float(), args)
			else:
				test_loss += mse(output, target.float()).data
			pred = output.data.squeeze()

			if(args.separate_trig):
				err = torch.mean(abs(output.float() - angle_to_separate_trig(target.float(), args)))
				pred = separate_trig_to_angle(pred, args)
			else:
				err = torch.mean(abs(output.float().squeeze() - target.float()))
			
			


		predn = pred.cpu().numpy()
		targn = target.cpu().numpy()

		if(all_pred is None):
			all_pred = predn
			all_targ = targn
			all_diff = difference(args,predn, targn)
		else:
			all_pred = np.hstack((all_pred, predn))
			all_targ = np.hstack((all_targ, targn))
			all_diff = np.hstack((all_diff, difference(args,predn, targn)))
	
		i_major+=args.batchSz

	all_pred = np.where(all_pred>360,all_pred%360,all_pred)

	test_stats(args, all_pred, all_targ, all_diff)

	test_loss /= len(dataloader) # loss function already averages over batch size
	nTotal = len(dataloader.dataset)
	incorrect = len(np.where(all_diff>5)[0])
	err = 100.*incorrect/nTotal
	print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%) > 5 deg\n'.format(
		test_loss, incorrect, nTotal, err))# ===============================================
	# distribution of error and label
	

	testF.write('{},{},{}\n'.format(epoch, test_loss, err))
	testF.flush()
	testF.close()


def adjust_opt(optAlg, optimizer, epoch):
	if optAlg == 'sgd':
		if epoch < 150: lr = 1e-1
		elif epoch == 150: lr = 1e-2
		elif epoch == 225: lr = 1e-3
		else: return

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


if __name__=='__main__':
	main()





