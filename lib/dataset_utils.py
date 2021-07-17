import torchvision.datasets as td
import torchvision.transforms as tt
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def analyze_data_code():
    '''
    分析pytorch data 相关代码
    :return:
    '''
    setup_seed(0)
    mnist_a = td.MNIST(root="/home/shicaiwei/data/domain_adda", transform=tt.ToTensor())
    mnist_b = td.MNIST(root="/home/shicaiwei/data/domain_adda", transform=tt.ToTensor())
    # 是迭代器,但是不能以next访问
    # a=mnist.next()

    # 这个对象的默认方法怎么访问?不像是transform 和model 不接受输入.

    # 不是生成器,因为可以无限次访问.
    # for data, label in mnist:
    #     print(label)
    # print(len(mnist))

    # SAMPLE
    # weights = []
    # num_samples = 0
    # for data, label in mnist:
    #     num_samples += 1
    #     if label == 0:
    #         weights.append(20)
    #     elif label == 1 or label == 2 or label == 4:
    #         weights.append(10)
    #     else:
    #         weights.append(1.6)
    # sampler = WeightedRandomSampler(weights,num_samples=len(mnist), replacement=True)

    # 相同打乱方式
    torch.manual_seed(0)  # ****
    loader_a = torch.utils.data.DataLoader(mnist_a, batch_size=32, shuffle=True)
    loader_b = torch.utils.data.DataLoader(mnist_b, batch_size=32, shuffle=True)

    # 获取迭代对象的迭代器
    a = loader_a.__iter__()
    b = loader_b.__iter__()
    print(len(a))

    # 访问迭代器
    lenght = 0
    while True:
        try:
            data_a, label_a = next(a)
            data_b, label_b = next(b)
            print(label_a - label_b)
            lenght += label_a.shape[0]
        except StopIteration as e:
            break
    print(lenght)

    for batch_idx, (data, target) in enumerate(loader_a):
        print(target.tolist())


if __name__ == '__main__':
    analyze_data_code()
