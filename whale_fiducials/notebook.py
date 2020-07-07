import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

from torch import nn, optim
import torch.nn.functional as F

IMG_SIZE = 128

data_dir = '../datasets/whales/'
data_csv = data_dir + 'keypoints.csv'
data_keypoints = pd.read_csv(data_csv)
# print(data_keypoints)


def show_keypoints(image, keypoints):
	plt.imshow(image)
	if len(keypoints):
		plt.scatter(keypoints[[0,2,4,6]], keypoints[[1,3,5,7]], s=24, marker ='.', c='r')

def show_images(df, indxs, ncols=5, figsize=(15,10), with_keypoints=True):
	'''
	Show images with keypoints in grids
	Args:
		df (DataFrame): data (M x N)
		idxs (iterators): list, Range, Indexes
		ncols (integer): number of columns (images by rows)
		figsize (float, float): width, height in inches
		with_keypoints (boolean): True if show image with keypoints
	'''
	plt.figure(figsize=figsize)
	nrows = len(indxs) // ncols + 1
	for i, idx in enumerate(indxs):
		# image = np.fromstring(df.loc[idx, 'Image'], sep=' ').astype(np.float32).reshape(-1, IMG_SIZE)
		filename = df.iloc[idx,0]
		print(data_dir+str(filename))
		image = cv2.imread(data_dir+str(filename))
		# print(image)
		print(df.iloc[0])
		# if with_keypoints:
		# 	try:
		# 		keypoints = df.loc[idx].drop('filename').values.astype(np.float32).reshape(-1, 2)
		# 	except: 
		# 		print('no filename column to drop')
		# else:
		# 	keypoints = []
		plt.subplot(nrows, ncols, i + 1)
		plt.title(f'Sample #{idx}')
		plt.axis('off')
		plt.tight_layout()
		show_keypoints(image, df.iloc[idx])
	plt.show()

# show_images(data_keypoints, range(10))

missing_any_data = data_keypoints[data_keypoints.isnull().any(axis=1)]
# print(missing_any_data.T.tail(10))
missing_data_indexes = missing_any_data.index

# show_images(data_keypoints, missing_data_indexes)

data_keypoints_nonnull = data_keypoints.dropna()
n = len(data_keypoints_nonnull)
np.random.seed(79)
permutation_order = np.random.permutation(n)
train_indices,val_indices,test_indices = np.split(permutation_order,[int(n*0.6),int(n*0.8)])

train_data = data_keypoints_nonnull.iloc[train_indices]
val_data = data_keypoints_nonnull.iloc[val_indices]
test_data = data_keypoints_nonnull.iloc[test_indices]
# print(train_data.shape)
# print(val_data.shape)
# print(test_data.shape)
print(train_data.iloc[0,0])
print(train_data.iloc[0,-1])








class WhaleDataset(Dataset):
	def __init__(self, dataframe, train=True, transform=None):
		self.dataframe=dataframe
		self.train=train
		self.transform = transform

	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self, idx):
		image = cv2.imread(data_dir+self.dataframe.iloc[idx,0],0)

		# if(self.train):
		keypoints = self.dataframe.iloc[idx,1:].values.astype(np.float32)

		sample = {'image':image, 'keypoints':keypoints}

		if(self.transform):
			sample = self.transform(sample)

		return sample

class Normalize(object):
	'''Normalize input images'''
	
	def __call__(self, sample):
		image, keypoints = sample['image'], sample['keypoints']
		
		return {'image': image / 255., # scale to [0, 1]
				'keypoints': keypoints}
		
class ToTensor(object):
	'''Convert ndarrays in sample to Tensors.'''

	def __call__(self, sample):
		image, keypoints = sample['image'], sample['keypoints']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		# image = image.reshape(1, IMG_SIZE, IMG_SIZE)
		image = transforms.ToTensor()(image)
		
		

		return {'image': image, 'keypoints': keypoints}


class Resize(object):

	def __init__(self, resize_to=128):
		self.resize_to=resize_to

	def __call__(self, sample):
		image, keypoints = sample['image'], sample['keypoints']

		image = transforms.ToPILImage()(image)
		image = transforms.Resize((self.resize_to, self.resize_to))(image)

		if keypoints is not None:
			keypoints = torch.from_numpy(keypoints)

		return {'image': image, 'keypoints': keypoints}

batch_size = 1
tsfm = transforms.Compose([Resize(),ToTensor(),Normalize()])


trainset = WhaleDataset(train_data,train=True,transform=tsfm)
valset = WhaleDataset(val_data,train=False,transform=tsfm)
testset = WhaleDataset(test_data,train=False,transform=tsfm)

# print(data_keypoints_nonnull.iloc[0])

train_loader = DataLoader(trainset, batch_size=batch_size)
val_loader = DataLoader(valset, batch_size=batch_size)
test_loader = DataLoader(testset, batch_size=batch_size)

from torch import nn, optim
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self, input_size, output_size, hidden_layers, drop_p =0.5):
		super(MLP, self).__init__()
		# hidden layers
		layer_sizes = [(input_size, hidden_layers[0])] \
					  + list(zip(hidden_layers[:-1], hidden_layers[1:]))
		self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) 
											for h1, h2 in layer_sizes])
		self.dropout = nn.Dropout(drop_p)
		self.output = nn.Linear(hidden_layers[-1], output_size)
		
	def forward(self, x):
		# flatten inputs
		x = x.view(x.shape[0], -1)
		for layer in self.hidden_layers:
			x = F.relu(layer(x))
			x = self.dropout(x)
		x = self.output(x)	
		return x

model = MLP(input_size=IMG_SIZE*IMG_SIZE, output_size=20, 
			hidden_layers=[128, 64], drop_p=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

def train(train_loader, valid_loader, model, criterion, optimizer, 
		  n_epochs=50, saved_model='model.pt'):
	'''
	Train the model
	
	Args:
		train_loader (DataLoader): DataLoader for train Dataset
		valid_loader (DataLoader): DataLoader for valid Dataset
		model (nn.Module): model to be trained on
		criterion (torch.nn): loss funtion
		optimizer (torch.optim): optimization algorithms
		n_epochs (int): number of epochs to train the model
		saved_model (str): file path for saving model
	
	Return:
		tuple of train_losses, valid_losses
	'''

	# initialize tracker for minimum validation loss
	try:
		model.load_state_dict(torch.load('model.pt'))
	except:
		print('failed to load state dict')
	valid_loss_min = None # set initial "min" to infinity

	train_losses = []
	valid_losses = []

	for epoch in range(n_epochs):
		# monitor training loss
		train_loss = 0.0
		valid_loss = 0.0

		###################
		# train the model #
		###################
		model.train() # prep model for training
		for batch in train_loader:
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			# forward pass: compute predicted outputs by passing inputs to the model
			output = model(batch['image'].to(device))
			print(output.data.int())
			print(batch['keypoints'].data.int())
			print()
			# calculate the loss
			loss = criterion(output, batch['keypoints'].to(device))
			# backward pass: compute gradient of the loss with respect to model parameters
			loss.backward()
			# perform a single optimization step (parameter update)
			optimizer.step()
			# update running training loss
			train_loss += loss.item()*batch['image'].size(0)

		######################	
		# validate the model #
		######################
		model.eval() # prep model for evaluation
		for batch in valid_loader:
			# forward pass: compute predicted outputs by passing inputs to the model
			output = model(batch['image'].to(device))
			# calculate the loss
			loss = criterion(output, batch['keypoints'].to(device))
			# update running validation loss 
			valid_loss += loss.item()*batch['image'].size(0)

		# print training/validation statistics 
		# calculate average Root Mean Square loss over an epoch
		train_loss = np.sqrt(train_loss/len(train_loader.dataset))
		valid_loss = np.sqrt(valid_loss/len(valid_loader.dataset))

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
			  .format(epoch+1, train_loss, valid_loss))

		# save model if validation loss has decreased
		if valid_loss_min and valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
				  .format(valid_loss_min, valid_loss))
			torch.save(model.state_dict(), saved_model)
			valid_loss_min = valid_loss
			
	return train_losses, valid_losses

train_losses, valid_losses = train(train_loader, val_loader, model,
								   criterion, optimizer, n_epochs=0, 
								   saved_model='model.pt')

def plot_RMSE(train_losses, valid_losses, y_max=50):
	plt.plot(train_losses, '--', linewidth=3, label='train')
	plt.plot(valid_losses, linewidth=3, label='val')
	plt.legend()
	plt.grid()
	plt.xlabel('Epoch')
	plt.ylabel('RMSE')
	plt.ylim((0, y_max))
	plt.savefig('graph.png')

plot_RMSE(train_losses, valid_losses, y_max=40)

def predict(data_loader, model):
	'''
	Predict keypoints
	Args:
		data_loader (DataLoader): DataLoader for Dataset
		model (nn.Module): trained model for prediction.
	Return:
		predictions (array-like): keypoints in float (no. of images x keypoints).
	'''
	
	model.eval() # prep model for evaluation
	with torch.no_grad():
		for i, batch in enumerate(data_loader):
			# forward pass: compute predicted outputs by passing inputs to the model
			output = model(batch['image'].to(device)).cpu().numpy()
			if i == 0:
				predictions = output
			else:
				predictions = np.vstack((predictions, output))
	return predictions

def view_pred_df(filenames, predictions, ncols=5, figsize=(15,10), image_ids=range(1,6)):
	'''
	Display predicted keypoints
	Args:
		columns (array-like): column names
		test_df (DataFrame): dataframe with ImageId and Image columns
		predictions (array-like): keypoints in float (no. of images x keypoints)
		image_id (array-like): list or range of ImageIds begin at 1
	'''

	cutoff = 5
	filenames=filenames[:cutoff]
	print(predictions.shape)
	predictions = predictions[:cutoff,:]
	print(predictions)

	plt.figure(figsize=figsize)
	nrows = len(filenames) // ncols + 1
	for i, (filename,keypoints) in enumerate(zip(filenames,predictions)):
		# image = np.fromstring(df.loc[idx, 'Image'], sep=' ').astype(np.float32).reshape(-1, IMG_SIZE)
		image = cv2.imread(data_dir+str(filename))
		print(keypoints.shape)
		plt.subplot(nrows, ncols, i + 1)
		# plt.title(f'Sample #{idx}')
		plt.axis('off')
		plt.tight_layout()
		show_keypoints(image, keypoints)
	plt.show()

# Load the minimum valuation loss model
model.load_state_dict(torch.load('model.pt'))
predictions = predict(test_loader, model)
# columns = train_data.drop('Image', axis=1).columns

view_pred_df(test_data.iloc[:,0].to_numpy(), predictions)