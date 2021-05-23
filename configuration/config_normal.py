import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decrease', type=str, default=None, help=' learning rate decay  ')
parser.add_argument('--lr_warmup', type=bool, default=False)
parser.add_argument('--mixup', type=bool, default=False, help='using mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--pretrain', type=bool, default=True)
parser.add_argument('--freeze', type=bool, default=False)


parser.add_argument('--model_name', type=str, default='mnist_cnn_best')
parser.add_argument('--log_name', type=str, default='.csv')
parser.add_argument('--train_dir', type=str,
                    default='/home/bbb//shicaiwei/data/liveness_data/CASIA_SURF_color_path/train')
parser.add_argument('--test_dir', type=str,
                    default='/home/bbb//shicaiwei/data/liveness_data/CASIA_SURF_color_path/test')

args = parser.parse_args()

args.mixup = False
