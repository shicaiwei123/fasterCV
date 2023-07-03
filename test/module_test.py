import torchvision.transforms as tt
import torch
import numpy as np
import sys

from lib.processing_utils import seed_torch

rotaion = tt.Compose([tt.RandomRotation(30)])


def transform_test():
    '''
    测试固定随机种子对transform 影响
    :return:
    '''
    from PIL import Image
    seed_torch(3)
    img_pil = Image.open("test_data/11.jpg")
    img_r = rotaion(img_pil)
    img_b = rotaion(img_pil)
    img_r.show()
    img_b.show()


def dropout_test():
    '''
    自己测试dropout 原理
    :return:
    '''
    def dropout(X, drop_prob):
        X = X.float()
        assert 0 <= drop_prob <= 1
        keep_prob = 1 - drop_prob
        # 这种情况下把全部元素都丢弃
        if keep_prob == 0:
            return torch.zeros_like(X)
        mask = (torch.rand(X.shape) < keep_prob).float()

        return mask * X / keep_prob

    W1 = torch.tensor(np.random.normal(0, 0.01, size=(16, 30)), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(30, requires_grad=True)

    X = torch.arange(16).view(1, 16)
    X = X.float()
    X = X.view(-1, 16)
    a = torch.matmul(X, W1) + b1
    b = dropout(a, 0.5)
    print(1)