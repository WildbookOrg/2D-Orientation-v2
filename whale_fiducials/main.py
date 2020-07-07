# -*- coding: utf-8 -*-
# regular imports
import os
import argparse
import shutil

# pytorch imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torch.nn as nn

# custom imports
from data import Data
from train import *
from example import *

def network_preprocessing(model, args):
	if(args.pretrain and not args.no_resume):
		print('Need option --no-resume in conjunction with --pretrian')
		exit(1)
	# choice to not resume
	if(args.no_resume):
		# change later when it matters
		
		# file exists -> confirm, delete, train from scratch
		if(os.path.exists(args.save_path)):
			shutil.rmtree(args.save_path)
			# =============================
			# i = input('Deleting save path.. continue (y/n) ')
			# if(i == 'y'):
			# 	shutil.rmtree(save_path) 
			# 	print('\t...Old state dict and stats files deleted')
			# else:
			# 	print('Cancelling program')
			# 	exit(1)
			# =============================
		# file does not exist, train from scratch
		# else:
		# 	print('No save path to delete')
	# choice to resume training
	else:
		# file found, load and use
		if(os.path.exists(args.pth_file)):
			print('State file found, loading...')
			model.load_state_dict(torch.load(args.pth_file))

		# no file found, exit
		else:
			print('No state dict file found... (use --no-resume to begin training from scratch)')
			exit(1)
	
	# if no file, the choice was to train from scratch
	if(not os.path.exists(args.save_path)):
		print('\t...Creating new working directory and empty files')

		os.makedirs(args.save_path, exist_ok=True)

	if(args.nClasses % 2 != 0):
		print('nClasses must be even')
		exit(1)

	# enable cuda
	if(args.cuda):
		model = model.cuda()

	# configure optimizer
	if args.opt == 'sgd':
		model.optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
	elif args.opt == 'adam':
		model.optimizer = optim.Adam(model.parameters(), weight_decay=args.lr)
	elif args.opt == 'rmsprop':
		model.optimizer = optim.RMSprop(model.parameters(), weight_decay=1e-4)

	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-resume', action='store_true', help='delete current trained weights and train from scratch')
	parser.add_argument('--pretrain', action='store_true', help='used with --no-resume, load wiped network with densenet pretrained values')
	parser.add_argument('--example', action='store_true', help='show a visualization of a sample of the test set')
	parser.add_argument('--percent-error', action='store_true', help='While calculating loss and error, use value from [0,1] instead of pixel distance')
	parser.add_argument('--test', action='store_true', help='test the current weights and plot some statistics')
	parser.add_argument('--debug', action='store_true', help='show debugging print statements')
	parser.add_argument('--augment', action='store_true', help='augment images used in training (perspective transform)')
	parser.add_argument('--batch-size', type=int, default=60, help='specify a batch size')
	parser.add_argument('--nClasses', type=int, default=4, help='specify the number of classes, for fluke tip regression this number must be even in the range of [2,20]')
	parser.add_argument('--nEpochs', type=int, default=50, help='specify a number of epochs')
	parser.add_argument('--resize-to', type=int, default=128, help='specify the image size used')
	parser.add_argument('--lr', type=float, default=1e-4, help='specify a learning rate')
	parser.add_argument('--type', default='regression_fluke_tips', help='specify a network type to be used with directory saving')
	parser.add_argument('--vinvine', action='store_true', help='some fun example that isnt expected to work')
	parser.add_argument('--train-bad-images', action='store_true', help='dont remove labeled poor images before training')
	parser.add_argument('--save-all-figures', action='store_true', help='apply the example function to each test image and save in a directory')
	parser.add_argument('--device', type=int, default=0, choices=tuple(range(torch.cuda.device_count())), help='specify the cuda device to run on')
	args = parser.parse_args()

	# args.nClasses = 20
	# args.type = 'regression_fluke_tips'
	args.opt = 'adam'
	args.cuda = torch.cuda.is_available()
	args.device = torch.device('cuda:{}'.format(int(args.device)) if args.cuda else 'cpu')

	args.save_path = 'work/densenet.%s.%s/'%(args.type,args.nClasses)
	args.save_file = '{}.{}.latest.pth'.format(args.type, args.nClasses)
	args.pth_file = os.path.join(args.save_path,args.save_file)

	
	# create model instance, add ending classifier
	try:
		model = torchvision.models.densenet161(pretrained=args.pretrain)
		model.classifier = nn.Linear(2208, args.nClasses)
	except:
		print('*model creation failed*')
		print('type this:\n\texport TORCH_HOME=~/.torch')
		exit(1)
	if(args.type.startswith('classification')):
		model.classifier = nn.Sequential(
			nn.Linear(2208, args.nClasses), 
			nn.LogSoftmax(dim=0)
		)
	model = network_preprocessing(model, args)
	model = model.to(args.device)

	if(args.vinvine):
		from PIL import Image
		image = Image.open('../datasets/vinvinewhale.jpg')
		image = image.resize((128,128))
		x = torchvision.transforms.functional.to_tensor(image)
		x.unsqueeze_(0)
		# print(x.shape)

		show_vinvine(model, x, args)
		exit(1)

	if(args.save_all_figures):
		args.example=True
		save_all_figures(model, Data(args, type='test'), args)
		exit(1)

	if(args.example):
		example_function(model, Data(args, type='test'), args)
		exit(1)

	data = {
		'train' : Data(args, type='train'),
		'val' : Data(args, type='val'),
		'test' : Data(args, type='test')
	}

	dataloaders = {
		'train' : DataLoader(data['train'],batch_size=args.batch_size,shuffle=True),
		'val' : DataLoader(data['val'],batch_size=args.batch_size,shuffle=True),
		'test' : DataLoader(data['test'],batch_size=args.batch_size,shuffle=False, drop_last=True)
	}

	datafiles = {
		'train' : open(args.save_path + 'train.csv', 'a'),
		'val' : open(args.save_path + 'val.csv', 'a'),
		'test' : open(args.save_path + 'test.csv', 'a')
	}

	# ===================================
	# do not delete
	# data['train'].review_bad_images()
	# exit(1)


	# set criterion
	if(args.type.startswith('classification')):
		loss_func = F.nll_loss
	if(args.type.startswith('regression')):
		loss_func = nn.MSELoss()

	if(args.test):
		test(args, model, dataloaders['test'], datafiles['test'], loss_func, 'test')
		exit(1)

	loss_history = {
		'train' : [[],[]],
		'val' : [[],[]],
		'save' : [[],[]]
	}
	
	# then train
	# best value loss calculated from the previous training session weights
	best_val_loss = train2(args, 0, model, dataloaders['val'], datafiles['val'], loss_func, 'val')
	print('val loss:', best_val_loss)
	print()
	for epoch in range(args.nEpochs):
		# adjust_opt(args.opt, model.optimizer, epoch)
		for phase in ['train','val']:
			# val_loss = train(args, epoch, model, dataloaders[phase], datafiles[phase], loss_func, phase)
			val_loss = train2(args, epoch, model, dataloaders[phase], datafiles[phase], loss_func, phase)
			loss_history[phase][0].append(val_loss)
			loss_history[phase][1].append(epoch)
			print(phase,'loss:',val_loss.item())
			print()

		if(val_loss<best_val_loss):
			print('Saving State Dict: new best loss is',val_loss)
			best_val_loss = val_loss
			loss_history['save'][0].append(val_loss)
			loss_history['save'][1].append(epoch)
			torch.save(model.state_dict(), args.pth_file)
			print()
		
	# close data files
	for key in datafiles.keys():
		datafiles[key].close()

	# plot the loss history
	c = ['b','r','black']
	for i,key in enumerate(loss_history.keys()):
		plt.plot(loss_history[key][0],loss_history[key][1],c=c[i])
	plt.show()

	