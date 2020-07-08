# -*- coding: utf-8 -*-
import torch
import numpy as np


def deg_to_radian(deg):
    return (deg * np.pi) / 180


def radian_to_deg(rad):
    return (rad * 180) / np.pi


def separate_trig_to_angle(t1):
    assert t1.size(1) == 2
    t1 = deg_to_radian(t1)
    t1[:, 0] = torch.acos(t1[:, 0])
    t1[:, 1] = torch.asin(t1[:, 1])
    t1 = radian_to_deg(t1)
    return torch.where(t1[:, 1] > 0, t1[:, 0], 360 - t1[:, 0])


def angle_to_separate_trig(t1):
    t1 = t1.reshape((t1.size(0), 1))
    t1 = deg_to_radian(t1)
    t_cos = torch.cos(t1)
    t_sin = torch.sin(t1)
    t1 = torch.cat((t_cos, t_sin), dim=1)
    t1 = radian_to_deg(t1)
    return t1


print('=================================================')

# target = torch.Tensor([  6, 153, 211, 325])
# print(target)


# manual
# target_trig = deg_to_radian(target)
# target_cos = torch.cos(target_trig)
# target_sin = torch.sin(target_trig)

# radian features are the same
# print('radian features:')
# print(target_cos.numpy())
# print(target_sin.numpy())
# print()

# target_cos = torch.acos(target_cos)
# target_sin = torch.asin(target_sin)
# target_cos = radian_to_deg(target_cos)
# target_sin = radian_to_deg(target_sin)
# print(target_cos.numpy())
# print(target_sin.numpy())


# print('----')
# functions
# target = angle_to_separate_trig(target)
# radian features are the same
# print('radian features')
# print(target[:,0].numpy())
# print(target[:,1].numpy())
# print()

# target = separate_trig_to_angle(target)
# print(target)


# print(separate_trig_to_angle(angle_to_separate_trig(target)).data)

target = torch.Tensor([78, 236, 111, 191, 32, 237, 277, 222])
print(angle_to_separate_trig(target))
output = torch.Tensor(
    [
        [-54.8898, 9.1113],
        [-46.2120, 22.0759],
        [-17.6196, 52.1301],
        [51.4009, 33.6172],
        [39.7999, -43.7839],
        [-54.2713, -25.5275],
        [-2.5029, -56.6218],
        [60.7458, 8.0755],
    ]
)
print(output)

angles = separate_trig_to_angle(output)
print(angles)
