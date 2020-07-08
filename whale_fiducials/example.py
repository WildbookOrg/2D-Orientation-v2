# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torchvision.utils as utils
import torch
import os


def example_function(model, data, args):
    model.eval()

    if args.batch_size > 8:
        args.batch_size = 8

    dataloader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    dataiter = iter(dataloader)

    images_normalized, images, keypoints = next(dataiter)

    with torch.no_grad():

        if args.cuda:
            data, target = images_normalized.cuda(), keypoints.cuda()
        output = model(data)

    output = output.cpu()

    if args.percent_error:
        imsize = images.shape[-2]
        output = output * imsize

    # print(output)
    # print(keypoints)

    ncols = 5
    figsize = (15, 10)

    plt.figure(figsize=figsize)
    nrows = args.batch_size // ncols + 1
    for i in range(keypoints.shape[0]):

        plt.subplot(nrows, ncols, i + 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(images[i])
        plt.scatter(keypoints[i, :][::2], keypoints[i, 1:][::2], s=24, marker='.', c='b')
        plt.scatter(output[i, :][::2], output[i, 1:][::2], s=24, marker='.', c='r')

    plt.show()

    # tensor_list = []
    # for i,image in enumerate(images):
    # 	image = np.array(image)
    # 	for x,y in zip(output[i,:][::2],output[i,1:][::2]):
    # 		cv2.circle(image, (int(x), int(y)), 3, (0,0,255),-1)
    # 	plt.imshow(image)
    # 	plt.show()

    # images = [transforms.ToPILImage()(image) for image in images]
    # images = [transforms.ToTensor()(image) for image in images]

    # grid = utils.make_grid(tensor_list)
    # plt.imshow(grid.numpy())
    # plt.axis('off')
    # plt.title(args.type+'\n'+str(angles.numpy())+'\n'+str(pred.int().numpy()))
    # plt.show()

    return


def save_all_figures(model, data, args):
    model.eval()

    save_dir = './results/{}.{}'.format(args.type, args.nClasses)

    # try not to mix files from different versions
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        i = input('There are files in this folder, proceed? (y/n)')
        if i != 'y':
            print('Cancelling')
            exit(1)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    dataloader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    i_major = 0
    for images_normalized, images, keypoints in dataloader:

        if args.cuda:
            data, target = images_normalized.cuda(), keypoints.cuda()
        output = model(data)

        if args.percent_error:
            imsize = images.shape[-2]
            output = output * imsize

        for i, image in enumerate(images):
            image = np.array(image)
            for x, y in zip(keypoints[i, :][::2], keypoints[i, 1:][::2]):
                cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
            for x, y in zip(output[i, :][::2], output[i, 1:][::2]):
                cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)
            print(i_major + i)

            plt.imshow(image)
            plt.savefig(save_dir + '/testfig{}.png'.format(i_major + i))

        i_major += args.batch_size

    return


def show_vinvine(model, data, args):
    model.eval()

    image = np.array(data)[0]
    print(image.shape)
    data = data.cuda()
    output = model(data)
    print(output)
    i = 0
    for x, y in zip(output[i, :][::2], output[i, 1:][::2]):
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    plt.imshow(image)
    plt.show()

    return
