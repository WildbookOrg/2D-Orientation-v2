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

from utils.test import *

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

	if(not args.hierarchy and args.separate_trig and args.type=='regression'):
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
			data, target = images_normalized.cuda(), angles.cuda()
	
	data = data.to(args.device)
	target = target.to(args.device)

	# grid = utils.make_grid(images)
	# plt.imshow(grid.numpy().transpose((1, 2, 0)))
	# plt.axis('off')
	# # plt.title(args.type+'\nPred: {}\nTrue: {}'.format(angles.detach().cpu().numpy(),pred.detach().cpu().int().numpy()))
	# plt.show()

	
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
	# °

def example_function_hierarchy(model, model_reg, args):
	if(args.batchSz>8):
		args.batchSz=8

	# get image data
	test_dataset = Data_turtles(dataType='test2020',experiment_type='example', args = args)
	testLoader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=True)
	test_iter = iter(testLoader)
	images_normalized, images, target= next(test_iter)
	if args.cuda:
			data = images_normalized.cuda()
	data = data.to(args.device)

	model.eval()
	model_reg.eval()

	with torch.no_grad():
		output = model(data).cpu()
		pred = output.data.max(1)[1]

		data = [transforms.ToPILImage()(image.cpu()*255) for image in data]	
		data = [transforms.functional.affine(image,-angle*(360//args.nClasses),(0,0),1,0) for image,angle in zip(data,pred)]
		data = torch.stack([transforms.ToTensor()(image).cuda() for image in data])/255
		grid = utils.make_grid(data.cpu()).permute(1,2,0).numpy()*255
		
		target_reg = target - pred*(360/args.nClasses)

		torch.where(target_reg<0,180+target_reg,target_reg)

		output_reg = model_reg(data).cpu()
		if(args.separate_trig):
			output_reg = separate_trig_to_angle(output_reg, args)
			
		data = [transforms.ToPILImage()(image.cpu()*255) for image in data]	
		data = [transforms.functional.affine(image,-angle,(0,0),1,0) for image,angle in zip(data,output_reg)]
		data = torch.stack([transforms.ToTensor()(image).cuda() for image in data])/255
		grid_reg = utils.make_grid(data.cpu()).permute(1,2,0).numpy()*255

	print(pred.data)
	print(target/(360/args.nClasses))
	print()
	print(output_reg.data)
	print(target_reg.data)
	print(target)
	
	plt.imshow(grid)
	plt.axis('off')
	plt.title("Classification ({} Class) - {}\nPred: {}\nTrue: {}".format(args.nClasses,args.animal,pred.data.numpy(),(target/(360/args.nClasses)).data.numpy()))
	plt.show()
	plt.imshow(grid_reg)
	plt.axis('off')
	plt.title('Regression - {}\nPred: {}\nTrue: {}'.format(args.animal,output_reg.int().data.numpy(),target_reg.data.numpy()))
	plt.show()



	# f, (ax,ax2) = plt.subplots(2, 1, sharex=True)
	# ax.imshow(grid)
	# ax2.imshow(grid)
	# ax.set_title("Classification ({} Class) - {}".format(args.nClasses,args.animal))
	# ax2.set_title("Regression Output - {}".format(args.nClasses,args.animal))
	# ax.axis('off')
	# ax2.axis('off')
	# plt.show() 

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
	if(args.hierarchy):
		train_dataset = Data_turtles(dataType = 'train2020', experiment_type='test', args = args)
		val_dataset = Data_turtles(dataType = 'val2020', experiment_type='test', args = args)
		test_dataset = Data_turtles(dataType='test2020', experiment_type='test', args = args)
	else:
		train_dataset = Data_turtles(dataType = 'train2020', experiment_type='train', args = args)
		val_dataset = Data_turtles(dataType = 'val2020', experiment_type='validation', args = args)
		test_dataset = Data_turtles(dataType='test2020', experiment_type='test', args = args)

	dataloaders = {
		'train' : DataLoader(train_dataset,batch_size=args.batchSz,shuffle=True,drop_last=True),
		'val' : DataLoader(val_dataset,batch_size=args.batchSz,shuffle=True,drop_last=True)
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
	parser.add_argument('--angle-range', type=int, default=360, help='angle range to train with; within [10,360]')
	parser.add_argument('--hierarchy', action='store_true',help='use the hierarchy model and combine classification and regression')
	args = parser.parse_args()

	if(args.hierarchy):
		args.type = 'hierarchy'


	# save a bunch of constants to avoid confusion later
	args.cuda = torch.cuda.is_available()
	args.save_path = 'work/densenet.%s.%s.%s%s/'%(args.animal,args.type,args.nClasses,'.pretrain' if args.pretrain else '')
	args.save_file = '{}.{}.latest.pth'.format(args.type, args.nClasses)
	args.pth_file = os.path.join(args.save_path,args.save_file)

	if(args.hierarchy):
		args.save_file_reg = '{}.{}.latest.pth'.format(args.type, 1)
		args.pth_file_reg = os.path.join(args.save_path,args.save_file_reg)

	arg_check(args)

	if(args.plot_loss_history):
		plot_loss_history(args)
		exit(1)



	# create model instance, add ending classifier
	# if hierarchy, this network is the classification network
	try:
		model = torchvision.models.densenet161(pretrained=args.pretrain)
	except:
		print('type this: \n\texport TORCH_HOME=~/.torch')
		exit(1)
	model.classifier = nn.Linear(2208, args.nClasses)
	if(args.type.startswith('classification') or args.hierarchy):
		model.classifier = nn.Sequential(
			nn.Linear(2208, args.nClasses), 
			nn.LogSoftmax(dim=0)
		)

	if(args.hierarchy):
		model_reg = torchvision.models.densenet161(pretrained=args.pretrain)
		c = 2 if(args.separate_trig) else 1
		model_reg.classifier = nn.Linear(2208, c)

	# resume training by loading state dict
	if not args.no_resume:
		model.load_state_dict(torch.load(args.pth_file))
		if(args.hierarchy):
			model_reg.load_state_dict(torch.load(args.pth_file_reg))
	model = model.to(args.device)
	if(args.hierarchy):
		model_reg = model_reg.to(args.device)

	
	if(args.save_all_figs):
		if(args.hierarchy):
			save_all_figures([model,model_reg],args)
		else:
			save_all_figures(model, args)
		exit(1)

	if args.example:
		if(args.hierarchy):
			example_function_hierarchy(model,model_reg,args)
		else:
			example_function(model, args)
		exit(1)

	# load files to track loss progress
	datafiles = get_text_files(args)

	if(args.type.startswith('classification') or args.hierarchy):
		loss_func = F.nll_loss
	elif(args.separate_trig):
		loss_func = trig_loss
	else:
		loss_func = mse

	if(args.hierarchy):
		if(args.separate_trig):
			loss_func_reg = trig_loss
		else:
			loss_func_reg = mse


	if(args.test):
		if(args.hierarchy):
			test_hierarchy(args, 1, model,model_reg,loss_func,loss_func_reg)
		else:
			test(args, 1, model)
		exit(1)
	# load dataloaders that hold each data set
	dataloaders = get_data_loaders(args)


	if args.cuda:
		model = model.cuda()
		if(args.hierarchy):
			model_reg.cuda()
	optimizer = get_optimizer(model, args)
	if(args.hierarchy):
		optimizer_reg = get_optimizer(model_reg,args)
	
	
	if(args.hierarchy):

		best_val_loss,best_val_loss_reg = train_hierarchy(args, 0, [model,model_reg], dataloaders['val'], [optimizer,optimizer_reg], datafiles['val'], [loss_func,loss_func_reg], 'val')
		print('best val loss:',best_val_loss,best_val_loss_reg)
		print()
		for epoch in range(1, args.nEpochs + 1):
			adjust_opt(args.opt, optimizer, epoch)
			adjust_opt(args.opt,optimizer_reg,epoch)
			for phase in ['train','val']:
				val_loss,val_loss_reg = train_hierarchy(args, epoch, [model,model_reg], dataloaders[phase], [optimizer,optimizer_reg], datafiles[phase], [loss_func,loss_func_reg], phase)
				
			if(val_loss<best_val_loss):
				print("Saving Class State Dict: new loss is",val_loss)
				best_val_loss = val_loss
				torch.save(model.state_dict(), args.pth_file)
			else:
				print('Class loss was:', val_loss)
				print()

			if(val_loss_reg<best_val_loss_reg):
				print("Saving Reg State Dicts: new loss is",val_loss_reg)
				best_val_loss_reg = val_loss_reg
				# torch.save(model.state_dict(), args.pth_file)
				torch.save(model_reg.state_dict(), args.pth_file_reg)
				
			else:
				print('Reg loss was:', val_loss_reg)
				print()
			
	else:
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
	with torch.set_grad_enabled(phase=='train'):
		for batch_idx, (data, target) in enumerate(dataloader):
			nProcessed += len(data)
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			# data, target = Variable(data), Variable(target)
			data = data.to(args.device)
			target = target.to(args.device)
			if(phase=='train'):
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

def train_hierarchy(args, epoch, net, dataloader, optimizer, datafile, loss_func, phase):
	net,net_reg = net 
	optimizer,optimizer_reg = optimizer
	loss_func,loss_func_reg = loss_func

	if(phase == 'train'):
		net.train()
		net_reg.train()
	else:
		net.eval()
		net_reg.eval()

	val_loss = 0
	val_loss_reg = 0
	nProcessed = 0
	incorrect = 0
	nTrain = len(dataloader.dataset)
<<<<<<< HEAD
	with torch.set_grad_enabled(phase=='train'):
		for batch_idx, (data, target) in enumerate(dataloader):

			nProcessed += len(data)
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data = data.to(args.device)
			target = target.to(args.device)

			if(phase=='train'):
				optimizer.zero_grad()
				optimizer_reg.zero_grad()
			
			# print()
			# =======================================
			# Classification step of hierarchy 
			output = net(data)

			# target for classification, so use bins
			loss = loss_func(output, target/(360/args.nClasses))

			pred = output.data.max(1)[1] # get the index of the max log-probability

			# print(output.data)

			err = torch.mean(abs(pred.float() - (target/(360/args.nClasses)).float()))
			val_loss += loss
			
			if(phase == 'train'):
				loss.backward()
				optimizer.step()

			# =======================================
			# Handle transition between levels
			# rotate data by output
			data = [transforms.ToPILImage()(image.cpu()*255) for image in data]	
			data = [transforms.functional.affine(image,-angle*(360//args.nClasses),(0,0),1,0) for image,angle in zip(data,pred)]
			data = torch.stack([transforms.ToTensor()(image).cuda() for image in data])/255
			# adjust target to newly rotated image
			target -= pred*(360/args.nClasses)
			torch.where(target<0,360-target,target)
			# =======================================
			# Regression step of hierarchy 
			output_reg = net_reg(data)
			loss_reg = loss_func_reg(output_reg.float(), target.float(), args)
			val_loss_reg += loss_reg
			pred = output_reg.data.squeeze()
			if(args.separate_trig):
				err = torch.mean(abs(output_reg.float() - angle_to_separate_trig(target.float(), args)))
			else:
				err = torch.mean(abs(output_reg.float().squeeze() - target.float()))
			if(phase == 'train'):
				loss_reg.backward()
				optimizer_reg.step()

			# print()
			# print('class loss:',loss.item())
			# print('class err:',err.item())
			# print('reg loss',loss_reg.item())
			# print('reg err:',err.item())
=======
	for batch_idx, (data, image, target) in enumerate(dataloader):
		print(0,'\t',torch.cuda.memory_allocated(args.device)//1000000)


		target_reg = target.cuda()
		target = torch.Tensor([int(angle/int(360/args.nClasses)) for angle in target]).long()

		# plt.imshow(data[0].permute(1,2,0)*255)
		# plt.title('train0')
		# plt.show()

		nProcessed += len(data)
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		# data, target = Variable(data), Variable(target)
		data = data.to(args.device)
		target = target.to(args.device)

		print(1,'\t',torch.cuda.memory_allocated(args.device)//1000000)


		optimizer.zero_grad()
		optimizer_reg.zero_grad()
		
		print()

		
		# =======================================
		# Classification step of hierarchy 
		output = net(data)
		print(2,'\t',torch.cuda.memory_allocated(args.device)//1000000)


		# print(output)
		# print(target)
			
		loss = loss_func(output, target)

		pred = output.data.max(1)[1] # get the index of the max log-probability
		# incorrect += pred.ne(target.data).cpu().sum()
		err = torch.mean(abs(pred.float() - target.float()))
		val_loss += loss
		
		if(phase == 'train'):
			loss.backward()
			optimizer.step()

		print(3,'\t',torch.cuda.memory_allocated(args.device)//1000000)

		# =======================================
		# Handle transition between levels
		# rotate data by output
		pred = output.data.max(1)[1] # get the index of the max log-probability
		# print(pred*(360//args.nClasses))


		# data = [transforms.ToPILImage()(image.cpu().detach()*255) for image in data]	
		# data = [transforms.functional.affine(image,-angle,(0,0),1,0) for image,angle in zip(data,pred)]
			
		# # plt.imshow(transforms.ToTensor()(data[0]).permute(1,2,0))
		# # plt.title('train1')
		# # plt.show()

		# data = torch.stack([transforms.ToTensor()(image) for image in data])
		
		# # plt.imshow(data[0].permute(1,2,0))
		# # plt.title('train2')
		# # plt.show()

		# data = data/255
		# data = data.cuda()

		# =======================================
		# Regression step of hierarchy 


		output_reg = net_reg(data)
		print(4,'\t',torch.cuda.memory_allocated(args.device)//1000000)
		# print(output_reg)
		# print(target_reg)
		loss_reg = loss_func_reg(output_reg.float(), target_reg.float(), args)
		
		val_loss_reg += loss_reg
		pred = output_reg.data.squeeze()

		if(args.separate_trig):
			err = torch.mean(abs(output_reg.float() - angle_to_separate_trig(target_reg.float(), args)))
		else:
			err = torch.mean(abs(output_reg.float().squeeze() - target_reg.float()))
		
		if(phase == 'train'):
			loss_reg.backward()
			optimizer_reg.step()

		print(5,'\t',torch.cuda.memory_allocated(args.device)//1000000)

		# print('class loss:',loss.item())
		# print('class err:',err.item())
		# print('reg loss',loss_reg.item())
		# print('reg err:',err.item())


		print()
>>>>>>> 8ec182cbc65b63c3a8d9d6cf038d3a7cfbec5f88

			partialEpoch = epoch + batch_idx / len(dataloader) - 1
			print(phase+': {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
				partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(dataloader),
				loss.data, err.item()))


<<<<<<< HEAD
			datafile.write('{},{},{}\n'.format(partialEpoch, loss.data, loss_reg.data))
			datafile.flush()	
=======
		datafile.write('{},{},{}\n'.format(partialEpoch, loss.item(), loss_reg.item()))
		datafile.flush()	
>>>>>>> 8ec182cbc65b63c3a8d9d6cf038d3a7cfbec5f88

		datafile.write('{},{}\n'.format(epoch,val_loss))
		print(epoch,val_loss)

	return val_loss,val_loss_reg

def test(args, epoch, net):
	test_dataset = Data_turtles(dataType='test2020', experiment_type='test', args = args)
	dataloader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=False, drop_last=True)
	testF = open(os.path.join(args.save_path, 'test.csv'), 'w')
	net.eval()
	incorrect = 0
	test_loss = 0
	all_pred = None
	all_targ = None
	all_diff = None
	i_major = 0
	with torch.no_grad():
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

def test_hierarchy(args, epoch, net, net_reg, loss_func, loss_func_reg):
	test_dataset = Data_turtles(dataType='test2020', experiment_type='test', args = args)
	dataloader = DataLoader(test_dataset,batch_size=args.batchSz,shuffle=False, drop_last=True)
	testF = open(os.path.join(args.save_path, 'test.csv'), 'w')
	net.eval()
	incorrect = 0
	test_loss = 0
	all_pred = None
	all_targ = None
	all_diff = None
	test_loss_reg = 0
	all_pred_reg = None
	all_targ_reg = None
	all_diff_reg = None
	i_major = 0
	with torch.no_grad():
		for data, display_image, target in dataloader:
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data = data.to(args.device)
			target = target.to(args.device)
			output = net(data)
			loss = loss_func(output, target/(360/args.nClasses))
			pred = output.data.max(1)[1] # get the index of the max log-probability
			err = torch.mean(abs(pred.float() - (target/(360/args.nClasses)).float()))
			test_loss += loss


			data = [transforms.ToPILImage()(image.cpu()*255) for image in data]	
			data = [transforms.functional.affine(image,-angle*(360//args.nClasses),(0,0),1,0) for image,angle in zip(data,pred)]
			data = torch.stack([transforms.ToTensor()(image).cuda() for image in data])/255
			
			b=360/args.nClasses # 45
			# print()
			# print('target angle:',target.data)
			# print('target/b:',(target/b).data)
			# print('pred*b:',pred.data,b)
			targ = (target/(360/args.nClasses)).cpu().numpy()
			
			target -= pred*(360/args.nClasses)
			# print('target - pred*b:',target.data)
			# print()
			torch.where(target<0,360-target,target)

			output_reg = net_reg(data)
			loss_reg = loss_func_reg(output_reg.float(), target.float(), args)
			test_loss_reg += loss_reg
			# pred_reg = output_reg.data.squeeze() # doesnt work with batch size 1
			if(args.separate_trig):
				pred_reg = separate_trig_to_angle(output_reg, args)
				err_reg = torch.mean(abs(output_reg.float() - angle_to_separate_trig(target.float(), args)))
			else:
				err_reg = torch.mean(abs(output_reg.float().squeeze() - target.float()))
			

						
			pred = pred.cpu().numpy()
			pred_reg = pred_reg.cpu().numpy()
			targ_reg = target.cpu().numpy()

			# print(pred, targ)


			if(all_pred is None):
				all_pred = pred
				all_targ = targ
				all_diff = difference(args, pred, targ)

				all_pred_reg = pred_reg
				all_targ_reg = targ_reg
				all_diff_reg = difference(args, pred_reg, targ_reg)


			else:
				all_pred = np.hstack((all_pred, pred))
				all_targ = np.hstack((all_targ, targ))
				all_diff = np.hstack((all_diff, difference(args, pred, targ)))
		

				all_pred_reg = np.hstack((all_pred_reg, pred_reg))
				all_targ_reg = np.hstack((all_targ_reg, targ_reg))
				all_diff_reg = np.hstack((all_diff_reg, difference(args, pred_reg, targ_reg)))
		
			i_major+=args.batchSz

		all_pred = np.where(all_pred>360,all_pred%360,all_pred)
		all_pred_reg = np.where(all_pred_reg>360,all_pred_reg%360,all_pred_reg)

		test_stats_hierarchy(args, all_pred, all_targ, all_diff, all_pred_reg, all_targ_reg, all_diff_reg)


		test_loss /= len(dataloader) # loss function already averages over batch size
		nTotal = len(dataloader.dataset)
		incorrect = len(np.where(all_diff>5)[0])
		err = 100.*incorrect/nTotal
		print('\nTest set: Class loss: {:.4f}, Reg loss: {}, Class err: {}, Reg err: {} \n'.format(
			test_loss, test_loss_reg, err, err_reg))


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





