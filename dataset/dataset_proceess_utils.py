import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb



class RandomHorizontalFlip_multi(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)

        if random.random() < self.p:
            for index in range(len(keys) - 1):
                value = sample[keys[index]]
                sample[keys[index]] = cv2.flip(value, 1)
            return sample
        else:
            return sample


class Resize_multi(object):

    def __init__(self, size):
        '''
        元组size,如(112,112)
        :param size:
        '''
        self.size = size

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            sample[keys[index]] = cv2.resize(value, self.size)
        return sample


class RondomRotion_multi(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)
        value_temp = sample[keys[0]]

        (h, w) = value_temp.shape[:2]
        (cx, cy) = (w / 2, h / 2)

        # 设置旋转矩阵
        angle = random.randint(-self.angle, self.angle)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0]) * 0.8
        sin = np.abs(M[0, 1]) * 0.8

        # 计算图像旋转后的新边界
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            sample[keys[index]] = cv2.warpAffine(value, M, (nw, nh))
        return sample


class RondomCrop_multi(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)
        value_temp = sample[keys[0]]

        h, w = value_temp.shape[:2]

        y = np.random.randint(0, h - self.size)
        x = np.random.randint(0, w - self.size)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            sample[keys[index]] = value[y:y + self.size, x:x + self.size, :]
        return sample


class Cutout_multi(object):
    '''
    作用在to tensor 之后
    '''

    def __init__(self, length=30):
        self.length = length

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)
        value_temp = sample[keys[0]]

        h, w = value_temp.shape[1], value_temp.shape[2]  # Tensor [1][2],  nparray [0][1]
        length_new = np.random.randint(1, self.length)
        y = np.random.randint(h - length_new)
        x = np.random.randint(w - length_new)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            value[y:y + length_new, x:x + length_new] = 0
            sample[keys[index]] = value
        return sample


class Normaliztion_multi(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            value = (value - 127.5) / 128  # [-1,1]
            sample[keys[index]] = value

        return sample


class ToTensor_multi(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W

        keys = sample.keys()
        keys = list(keys)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            value = value.transpose((2, 0, 1))
            sample[keys[index]] = value

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            value = np.array(value)
            value = torch.from_numpy(value.astype(np.float)).float()
            # print(value.type())
            sample[keys[index]] = value

        return sample
