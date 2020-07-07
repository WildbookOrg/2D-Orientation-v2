# -*- coding: utf-8 -*-
# regular imports 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# pytorch imports
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# custom imports


# custom transforms 
class HorizontalFlip(object):

	def __init__(self, randint):
		self.randint = randint

	def __call__(self, image):
		if(self.randint < 0.5):
			return F.hflip(image)
		return image

class Perspective(object):
	"""Performs Perspective transformation of the given PIL Image randomly with a given probability.
	Args:
		interpolation : Default- Image.BICUBIC
		p (float): probability of the image being perspectively transformed. Default value is 0.5
		distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
	"""

	def __init__(self, randint, distortion_scale=0.05, p=0.5, interpolation=Image.BICUBIC):
		self.randint = randint
		self.p = p
		self.interpolation = interpolation
		self.distortion_scale = distortion_scale

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be Perspectively transformed.
		Returns:
			PIL Image: Random perspectivley transformed image.
		"""
		
		if self.randint < self.p:
			width, height = img.size
			startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
			return self.perspective(img, startpoints, endpoints, self.interpolation)
		return img

	def get_params(self, width, height, distortion_scale):
		"""Get parameters for ``perspective`` for a random perspective transform.
		Args:
			width : width of the image.
			height : height of the image.
		Returns:
			List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
			List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
		"""
		half_height = int(height / 2)
		half_width = int(width / 2)
		topleft = (np.random.uniform(0, int(distortion_scale * half_width)),
				   np.random.uniform(0, int(distortion_scale * half_height)))
		topright = (np.random.uniform(width - int(distortion_scale * half_width) - 1, width - 1),
					np.random.uniform(0, int(distortion_scale * half_height)))
		botright = (np.random.uniform(width - int(distortion_scale * half_width) - 1, width - 1),
					np.random.uniform(height - int(distortion_scale * half_height) - 1, height - 1))
		botleft = (np.random.uniform(0, int(distortion_scale * half_width)),
				   np.random.uniform(height - int(distortion_scale * half_height) - 1, height - 1))
		startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
		endpoints = [topleft, topright, botright, botleft]
		
		self.startpoints = startpoints
		self.endpoints = endpoints

		return startpoints, endpoints

	def __repr__(self):
		return self.__class__.__name__ + '(p={})'.format(self.p)

	def perspective(self, img, startpoints, endpoints, interpolation=Image.BICUBIC):
		"""Perform perspective transform of the given PIL Image.
		Args:
			img (PIL Image): Image to be transformed.
			startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image
			endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
			interpolation: Default- Image.BICUBIC
		Returns:
			PIL Image:  Perspectively transformed Image.
		"""
		matrix = []
		for p1, p2 in zip(endpoints, startpoints):
			matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
			matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

		# print(start)
		# print(end)
		# print(matrix)

		A = torch.tensor(matrix, dtype=torch.float)
		B = torch.tensor(startpoints, dtype=torch.float).view(8)
		res = torch.gels(B, A)[0]
		coeffs = res.squeeze_(1).tolist()
		
		return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation)

class RandomCrop(object):
	def __init__(self, randint):
		self.randint = randint

	def __call__(self, image):
		'TODO'


class Data():
	def __init__(self, args, type='train'):
		self.args = args

		# load and clean images
		self.dataDir = '../datasets/whale_keypoints/'
		keypoints = np.genfromtxt(self.dataDir+'keypoints.csv', delimiter=',', skip_header=True, dtype=str)
		filenames = keypoints[:,0]
		keypoints = keypoints[:,1:]

		bad_images = np.genfromtxt('./bad_images.txt',delimiter=',',dtype=int)[:,0]
		self.bad_keypoints = keypoints[bad_images]
		self.bad_filenames = filenames[bad_images]
		keypoints = np.delete(keypoints, bad_images, 0)
		filenames = np.delete(filenames, bad_images, 0)

		bad_rows = np.unique(np.where(keypoints=='')[0])
		keypoints = np.delete(keypoints, bad_rows, 0)
		filenames = np.delete(filenames, bad_rows, 0)
		self.keypoints = keypoints.astype(float)
		self.filenames = filenames.astype(str)

		# ===========================
		# split into train, val, and test set
		if(type):
			self.type = type

		n = len(self.filenames)
		np.random.seed(79)
		permutation_order = np.random.permutation(n)
		train_indices,val_indices,test_indeces = np.split(permutation_order,[int(n*0.6),int(n*0.8)])

		assert(len(filenames)==len(keypoints))
		self.training_filenames = filenames[train_indices]
		self.training_keypoints = keypoints[train_indices]
		self.validation_filenames = filenames[val_indices]
		self.validation_keypoints = keypoints[val_indices]
		self.testing_filenames = filenames[test_indeces]
		self.testing_keypoints = keypoints[test_indeces]
		assert(len(self.training_filenames)==len(self.training_keypoints))

		# =============
		# use func if need to recalculate means/stds
		# self.get_mean_std()
		# print()
		# print(self.type)
		# print(self.means)
		# print(self.stds)
		# =============

		if(self.type.startswith('train')):
			self.means = [154.90263337943188, 144.07401797283885, 134.5996525401871]
			self.stds = [59.57422763, 58.71576831, 58.66761082]
		if(self.type.startswith('val')):
			self.means = [156.5270606905154, 145.76389668612032, 135.9003840081959]
			self.stds = [58.5250723,  57.96740715, 58.28005897]
		if(self.type.startswith('test')):
			self.means = [154.29151350794425, 145.80935633853818, 138.96590502742262]
			self.stds = [60.25371677, 59.90365963, 60.15483666]


	def __getitem__(self, index):
		# pull from corresponding dataset
		if(self.type.startswith('train')):
			im = cv2.imread(self.dataDir+self.training_filenames[index])
			keypoints = self.training_keypoints[index].astype(float)
		if(self.type.startswith('val')):
			im = cv2.imread(self.dataDir+self.validation_filenames[index])
			keypoints = self.validation_keypoints[index].astype(float)
		if(self.type.startswith('test')):
			im = cv2.imread(self.dataDir+self.testing_filenames[index])
			keypoints = self.testing_keypoints[index].astype(float)

		show = False
		
		if(self.args.nClasses == 4):
			# just the x,y of the two tips
			keypoints = keypoints[[0,1,8,9]]
		else:
			keypoints = keypoints[:self.args.nClasses]
			
		if(self.args.augment and self.type.startswith('train')):
			image_normalized, augmented_keypoints = self.augment_image_train(im, keypoints)
		else:
			image_normalized, original, augmented_keypoints = self.augment_image_test(im, keypoints)
			I = np.array(original)
		

		

		if(show):
			self.show_keypoints(I, augmented_keypoints)

		if(self.args.example):
			return image_normalized, I, augmented_keypoints
		
		return image_normalized, augmented_keypoints

	def augment_image_train(self, image, keypoints):
		# original = image.copy()
		h,w = image.shape[:2]
		hflip_randint = np.random.uniform()

		# force perspective transform for everything
		persp_randint = 0. # np.random.uniform() 
	

		pt = Perspective(persp_randint)

		tsfm = transforms.Compose([
					transforms.ToPILImage(),
					transforms.Resize((self.args.resize_to,self.args.resize_to)),
					HorizontalFlip(hflip_randint),
					transforms.RandomRotation(10),
					pt,
					# RandomCrop(self.resize_to//1.05),
					# transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
					# transforms.ToTensor()
			])


		plt.subplot(1, 2, 1)
		plt.axis('off')
		plt.tight_layout()
		plt.imshow(image)
		image = tsfm(image)
		plt.subplot(1, 2, 2)
		plt.axis('off')
		plt.tight_layout()
		plt.imshow(image)
		plt.show()

		
		keypoints[list(range(0,self.args.nClasses,2))] = keypoints[list(range(0,self.args.nClasses,2))]*self.args.resize_to/w
		keypoints[list(range(1,self.args.nClasses,2))] = keypoints[list(range(1,self.args.nClasses,2))]*self.args.resize_to/h

		src = np.array(pt.startpoints).astype(np.float32)
		dst = np.array(pt.endpoints).astype(np.float32)

		print(src)
		print(dst)

		M = cv2.getAffineTransform(src, dst)
		keypoints = cv2.perspectiveTransform(keypoints, M)
		print(keypoints)

		return image, keypoints

	def augment_image_test(self, image, keypoints):
		h,w = image.shape[:2]
		I = transforms.ToPILImage()(image)
		I = transforms.Resize((self.args.resize_to,self.args.resize_to))(I)
		image_normalized = transforms.ToTensor()(I)/255.0	
		keypoints[list(range(0,self.args.nClasses,2))] = keypoints[list(range(0,self.args.nClasses,2))]*self.args.resize_to/w
		keypoints[list(range(1,self.args.nClasses,2))] = keypoints[list(range(1,self.args.nClasses,2))]*self.args.resize_to/h
		return image_normalized, I, keypoints





	def __len__(self):
		# return length of corresponding dataset
		if(self.type.startswith('train')):
			return len(self.training_filenames)
		if(self.type.startswith('val')):
			return len(self.validation_filenames)
		if(self.type.startswith('test')):
			return len(self.testing_filenames)

	def get_mean_std(self):
		print('getting means and stds')
		# pull from corresponding dataset
		if(self.type.startswith('train')):
			filenames = self.training_filenames
		if(self.type.startswith('val')):		
			filenames = self.validation_filenames
		if(self.type.startswith('test')):
			filenames = self.testing_filenames

		N = len(filenames)
		means = [0,0,0]
		stds = [0,0,0]
		for i,fname in enumerate(filenames):
			I = cv2.imread(self.dataDir+fname)
			for channel in range(3):
				means[channel] += np.mean(I[:,:,channel])/N
				stds[channel] += np.std(I[:,:,channel])**2/N
		stds = np.sqrt(stds)
		self.means = means
		self.stds = stds
		print('finished getting means and stds')


	def show_keypoints(self, image, keypoints, show_keypoints=True):
		

		plt.imshow(image)
		if(show_keypoints):
			plt.scatter(keypoints[list(range(0,self.nClasses,2))], keypoints[list(range(1,self.nClasses,2))], s=24, marker ='.', c='r')
		plt.show()

	def review_good_images(self):
		progress_file = open('progress.txt','a+')
		for index in range(0,len(self.training_filenames)):
			try:
				im = cv2.imread(self.dataDir+self.training_filenames[index])
				# iterate 2 keypoint values at a time (x,y)
				for x,y in zip(self.training_keypoints[index,:][::2],self.training_keypoints[index,1:][::2]):
					cv2.circle(im, (int(float(x)), int(float(y))), 5, (0,0,255),-1)
				cv2.imshow('whale',im)
				k = cv2.waitKey(0)
				if(k==98): # b
					print(self.training_filenames[index])
					progress_file.write(str(index)+','+self.training_filenames[index]+'\n')
				if(k==113): # q
					break
				if(k==99): # c
					print('index:',index)
					progress_file.write(str(index)+'\n')
			except:
				progress_file.write('Failed at %s,%s\n'%(str(index),self.filenames[index]))
				print(self.training_keypoints[index])

	def review_bad_images(self):
		for index in range(len(self.bad_filenames)):
			try:
				im = cv2.imread(self.dataDir+self.bad_filenames[index])
				for x,y in zip(self.bad_keypoints[index,:][::2],self.bad_keypoints[index,1:][::2]):
					cv2.circle(im, (int(float(x)), int(float(y))), 5, (0,0,255),-1)
				cv2.imshow('whale',im)
				k = cv2.waitKey(0)
				if(k==113): # q
					break
				if(k==99): # c
					print('Filename:',self.bad_filenames[index])
			except:
				print('Failed at',self.bad_filenames[index])