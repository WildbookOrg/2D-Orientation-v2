# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable


def mse(t1, t2):
    diff = torch.abs(t1.squeeze() - t2.squeeze())
    m = torch.mean(diff, dim=0)
    return m * m


def mse2d(t1, t2):

    # print(t1.shape,t2.shape)
    diff = torch.abs(t1.squeeze() - t2.squeeze())
    # print(diff.shape)
    m = torch.mean(diff, dim=0)
    # print(m.shape)
    m = m * m
    return m


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def print_statistics(epoch, nProcessed, nTrain, batch_idx, loss, err, datafile, phase):
    partialEpoch = epoch + nProcessed / nTrain
    print(
        phase
        + ': {:.2f} [{:.2f}/{:.2f} ({:.2f}%)]\tLoss: {:.2f}\tError: {:.2f}'.format(
            partialEpoch,
            nProcessed,
            nTrain,
            100.0 * nProcessed / nTrain,
            (loss.item()),
            err,
        )
    )

    datafile.write('{:.2f},{:.2f},{:.2f}\n'.format(partialEpoch, (loss.item()), err))


def train(args, epoch, model, dataloader, datafile, loss_func, phase):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    val_loss = torch.Tensor([0.0, 0.0, 0.0, 0.0]).cuda()
    nProcessed = 0
    nTrain = len(dataloader.dataset)
    with torch.set_grad_enabled(phase == 'train'):
        for batch_idx, (data, target) in enumerate(dataloader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)
            nProcessed += len(data)
            data = data.to(args.device)
            target = target.to(args.device)

            model.optimizer.zero_grad()
            output = model(data)

            if args.debug:
                print(output)
                print(target)
                print()

            if args.type.startswith('classification'):
                loss = F.nll_loss(output, target)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                incorrect = pred.ne(target.data).cpu().sum()
                err = 100.0 * incorrect / len(data)
            if args.type.startswith('regression'):
                loss = mse2d(output.float(), target.float())
                err = torch.mean(abs(output.float().squeeze() - target.float()))
                l1 = mse(output.float()[:, 0], target.float()[:, 0])
                l2 = mse(output.float()[:, 1], target.float()[:, 1])
                l3 = mse(output.float()[:, 2], target.float()[:, 2])
                l4 = mse(output.float()[:, 3], target.float()[:, 3])

            if phase == 'train':

                l1.backward(retain_graph=True)
                l2.backward(retain_graph=True)
                l3.backward(retain_graph=True)
                l4.backward()
                model.optimizer.step()

            val_loss += torch.mean(loss)

            # val_loss += loss.data
            print_statistics(
                epoch,
                nProcessed,
                nTrain,
                batch_idx,
                torch.mean(loss.data),
                err,
                datafile,
                phase,
            )
    return val_loss


def train2(args, epoch, model, dataloader, datafile, loss_func, phase):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    val_loss = 0.0
    nProcessed = 0
    nTrain = len(dataloader.dataset)
    with torch.set_grad_enabled(phase == 'train'):
        for batch_idx, (data, target) in enumerate(dataloader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)
            nProcessed += len(data)
            data = data.to(args.device)
            target = target.to(args.device)

            if args.percent_error:
                assert data.shape[-1] == data.shape[-2]
                imsize = data.shape[-1]
                target /= imsize

            model.optimizer.zero_grad()
            output = model(data)

            if args.debug:
                print(output.int()[0].data)
                print(target.int()[0].data)
                print()

            if args.type.startswith('regression'):
                loss = loss_func(output.float(), target.float())
                err = torch.mean(abs(output.float().squeeze() - target.float()))

            if phase == 'train':
                loss.backward()
                model.optimizer.step()

            val_loss += torch.mean(loss)

            print_statistics(
                epoch,
                nProcessed,
                nTrain,
                batch_idx,
                torch.mean(loss.data),
                err,
                datafile,
                phase,
            )

    return val_loss


def test(args, model, dataloader, datafile, loss_func, phase):
    model.eval()

    test_loss = 0.0
    nProcessed = 0
    nTrain = len(dataloader.dataset)

    differences = None

    with torch.set_grad_enabled(False):
        for batch_idx, (data, target) in enumerate(dataloader):
            if args.cuda:
                data, target = data.cuda(), target.cuda().float()
            nProcessed += len(data)
            data = data.to(args.device)
            target = target.to(args.device)

            if args.percent_error:
                assert data.shape[-1] == data.shape[-2]
                imsize = data.shape[-1]
                target /= imsize

            model.optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output.float(), target.float())
            err = torch.mean(abs(output.float().squeeze() - target.float()))
            test_loss += torch.mean(loss)
            print_statistics(
                0,
                nProcessed,
                nTrain,
                batch_idx,
                torch.mean(loss.data),
                err,
                datafile,
                phase,
            )

            diff = output - target

            if differences is None:
                differences = diff
            else:
                differences = torch.cat((differences, diff), dim=0)

    # differences = differences*128

    import matplotlib.pyplot as plt
    import numpy as np

    x1 = differences[:, 0].cpu().numpy()
    y1 = differences[:, 1].cpu().numpy()
    x2 = differences[:, 2].cpu().numpy()
    y2 = differences[:, 3].cpu().numpy()

    print('left x mean: ', np.mean(np.abs(x1)))
    print('left y mean: ', np.mean(np.abs(y1)))
    print('right x mean: ', np.mean(np.abs(x2)))
    print('right y mean: ', np.mean(np.abs(y2)))

    print('left x std: ', np.std(np.abs(x1)))
    print('left y std: ', np.std(np.abs(y1)))
    print('right x std: ', np.std(np.abs(x2)))
    print('right y std: ', np.std(np.abs(y2)))

    """
	left x mean:  1.6042521
	left y mean:  2.04893
	right x mean:  1.470349
	right y mean:  1.6106945
	left x std:  1.8232398
	left y std:  2.324086
	right x std:  1.8327305
	right y std:  2.8957098
	"""

    # plot histogram
    plt.hist(
        x1,
        label='Left Tip X',
        alpha=0.4,
        bins=np.linspace(np.min(x1), np.max(x1), num=50),
    )
    plt.hist(
        x2,
        label='Right Tip X',
        alpha=0.4,
        bins=np.linspace(np.min(x2), np.max(x2), num=50),
    )
    plt.hist(
        y1,
        label='Left Tip Y',
        alpha=0.4,
        bins=np.linspace(np.min(x2), np.max(x2), num=50),
    )
    plt.hist(
        y2,
        label='Right Tip Y',
        alpha=0.4,
        bins=np.linspace(np.min(x2), np.max(x2), num=50),
    )
    plt.title('Error Histograms of Tip Coordinates')
    plt.xlabel('Difference between Predicted and True locations')
    plt.ylabel('Frequency in Test Set')
    plt.legend()
    plt.show()

    n = len(x1)

    plt.title('X and Y Error Respective of Axis')
    plt.xlabel('Error X')
    plt.ylabel('Error Y')
    plt.grid()
    plt.scatter(
        x1,
        y1,
        s=80,
        alpha=0.6,
        label=('Left Tip X & Y'),
        facecolors='none',
        edgecolors='r',
    )
    plt.scatter(
        x2,
        y2,
        s=80,
        alpha=0.3,
        label=('Right Tip X & Y'),
        facecolors='none',
        edgecolors='b',
    )
    plt.scatter(
        x1[: n // 5], y1[: n // 5], s=80, alpha=0.2, facecolors='none', edgecolors='r'
    )
    plt.legend()

    plt.show()
