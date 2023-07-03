
'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_cbam import resnet18_cbam
from src.xxx_dataloader import xxx_dataloader
from configuration.config_normal import args
import torch
import torch.nn as nn
from lib.model_develop_utils import train_base
from lib.processing_utils import get_file_list
import torch.optim as optim

import cv2
import numpy as np
import datetime
import random

'''
TO DO:
debug resnet 预训练参数加载
debug resnet 底层层数设计,特别是make layer 层,弄明白模型参数重载的原理
'''


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main(args):


    train_loader = xxx_dataloader(train=True, args=args)
    test_loader = xxx_dataloader(train=False, args=args)

    seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    model = resnet18_cbam(args, pretrained=True)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
    #                        weight_decay=args.weight_decay)

    args.retrain = False
    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
               test_loader=test_loader,
               args=args)


if __name__ == '__main__':

    deeppix_main(args=args)
