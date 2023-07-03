'''保留有意义的测试代码'''
import torchvision.transforms as tt
import torch
import numpy as np
rotaion = tt.Compose([tt.RandomRotation(30)])
from multiprocessing import Process
import os
import cv2
from lib.processing_utils import makedir,get_file_list,recut_face_with_landmarks
def transform_test():
    from PIL import Image

    img_pil = Image.open(
        "/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/31.jpg").convert(
        'RGB')
    img_r = rotaion(img_pil)
    img_b = rotaion(img_pil)
    img_r.show()
    img_b.show()


def dropout_test():
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


def pool_test():
    from models.surf_baseline import SURF_Baseline
    from configuration.config_baseline_multi import args
    import torch.nn.functional as tnf

    a = torch.tensor([np.float32(x) for x in range(9)])
    a = a.reshape(3, -1)
    a = torch.unsqueeze(a, 0)
    b = tnf.adaptive_avg_pool2d(a, (2, 2))
    print(b)


def gather_unique_test():
    indices = torch.tensor([2, 3, 1, 2])
    x = np.reshape(range(16), (4, 4))
    print(x.shape)
    indices_unique, index = np.unique(indices, return_index=True)
    zip_indices_idnex = zip(indices_unique, index)
    sort_zip = sorted(zip_indices_idnex, key=(lambda x: x[1]))
    indices_unique = [i[0] for i in sort_zip]
    print(indices_unique)

    a = np.take(x, axis=0, indices=indices_unique)
    print(a)


def get_landmarks_face(start, end):
    data_path = "/home/data/shicaiwei/oulu/Test_face_normal"
    save_dir = "/home/data/shicaiwei/oulu/Test_face_landmarks"
    video_list = os.listdir(data_path)
    video_list.sort()
    video_list = video_list[start:end]
    print(start)
    for video in video_list:
        video_path = os.path.join(data_path, video)
        img_path_list = get_file_list(video_path)
        img_path_list.sort()

        save_dir = os.path.join(save_dir, video)
        makedir(save_dir)
        for path in img_path_list:
            print(path)
            img = cv2.imread(path)
            landmarks_face = recut_face_with_landmarks(img)
            img_name = path.split('/')[-1]
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, landmarks_face)

        save_dir = "/home/data/shicaiwei/oulu/Test_face_landmarks"


def multimodal_test():
    process_list = []
    for i in range(20):  # 开启5个子进程执行fun1函数
        p = Process(target=get_landmarks_face, args=(i * 90, i * 90 + 90,))  # 实例化进程对象
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

