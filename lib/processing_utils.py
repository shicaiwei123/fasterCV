import os
import numpy as np
import torch
import csv
import cv2
import dlib
import random
import h5py
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
import scipy.io as scio
import mat73

from sklearn.metrics import roc_curve


def get_err_threhold(fpr, tpr, threshold):
    RightIndex = (tpr + (1 - fpr) - 1)
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    # print(err, best_th)
    return err, best_th


class FaceDection(object):
    '''
    提供多种人脸检测方法的实现
    输入人脸,选择检测模式,输出人脸图像
    '''

    def __init__(self, model_name, face_max=True):
        '''
        :param model_name: 选择人脸检测的模型
        :param face_max: 返回的人脸数目,是最大的人脸还是所有人脸.目前该功能还未实现.默认只返回最大的人脸
        '''
        self.model_name = model_name
        self.face_max = face_max

        if model_name == "CAFFE":
            modelFile = "../lib/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "../lib/deploy.prototxt"
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
            self.model = net
        elif model_name == "TF":
            modelFile = "../lib/opencv_face_detector_uint8.pb"
            configFile = "../lib/opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
            self.model = net
        else:
            self.face_cascade = cv2.CascadeClassifier('../lib/haarcascade_frontalface_default.xml')

        self.conf_threshold = 0.7

    def face_detect(self, img, display=False):
        '''
        输入人脸,返回人脸照片
        :param img:
        :param display:
        :return:
        '''

        if self.model_name == "cv2":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))

            # 分析
            if len(faces) == 0:
                return None
            else:

                # 求最大人脸
                face_max = [0, 0, 0, 0]
                for i in range(len(faces)):
                    face = faces[i]
                    if face[2] > face_max[2]:
                        face_max = face

                # 人脸截取
                left = face_max[0]
                top = face_max[1]
                right = left + face_max[2]
                bottom = top + face_max[2]
                face_img = img[top:bottom, left:right]
        else:

            frameOpencvDnn = img.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            img_mean = np.mean(frameOpencvDnn, (0, 1))

            # 数据预处理
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), img_mean, False, False)

            self.model.setInput(blob)
            detections = self.model.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            if display:
                cv2.imshow("frame", frameOpencvDnn)
                cv2.waitKey(0)

            # 如果检测到多个人脸,取最大的人脸
            if len(bboxes) > 1:
                bbox_max = [0, 0, 0, 0]
                bbox_max_len = bbox_max[2] - bbox_max[0]
                for bbox in bboxes:
                    if (bbox[2] - bbox[0]) > bbox_max_len:
                        bbox_max = bbox
                        bbox_max_len = bbox_max[2] - bbox_max[0]

                face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]
            elif len(bboxes) == 0:
                '''检测不到人脸'''
                return None
            else:
                bbox_max = bboxes[0]
                face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]

        return face_img


class LandmarksDetection(object):
    '''
    提供了多种人脸关键点检测算法的实现
    输入的最好是单个人脸图像
    '''

    def __init__(self):
        PREDICTOR_PATH = '../lib/shape_predictor_68_face_landmarks.dat'
        self.a_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def landmarks_detect(self, img_rgb, display):
        '''
        输入人脸,检测关键点,最好是人脸检测之后的人脸图像.
        :param img:
        :return:
        '''
        img_shape = img_rgb.shape
        # 类型转变，opencv_to_dlib
        x1 = 0
        y1 = 0
        x2 = x1 + img_shape[0]
        y2 = y1 + img_shape[1]
        rect = dlib.rectangle(x1, y1, x2, y2)

        img_key = img_rgb.copy()
        predictor = self.a_predictor
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        points_keys = []

        # 特征点检测,只取第一个,也就是最大的一个
        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])

        # 特征点提取,标注
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            points_keys.append([point[0, 0], point[0, 1]])
            cv2.circle(img_key, pos, 2, (255, 0, 0), -1)

        if display:
            cv2.imshow("landmark", img_key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("landmark display")
        self.a_landmark = points_keys
        return points_keys


def recut_face_with_landmarks(img, display=False):
    '''
    检测人脸关键点,利用关键点裁剪人脸,
    :param origin_face_dir:
    :param target_face_dir:
    :return:
    '''
    landmarks_detector = LandmarksDetection()
    x_data = np.zeros(27)
    y_data = np.zeros(27)

    h, w = img.shape[0], img.shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)
    face_landmarks = landmarks_detector.landmarks_detect(img, display=display)
    face_landmarks = np.array(face_landmarks)
    x_temp = face_landmarks[0:17, 0]
    y_temp = face_landmarks[0:17, 1]
    x_data[0:17] = x_temp
    y_data[0:17] = y_temp
    for i in range(5):
        a = face_landmarks[26 - i][0]
        x_data[17 + i] = face_landmarks[26 - i][0]
        y_data[17 + i] = face_landmarks[26 - i][1]

    for i in range(5):
        x_data[22 + i] = face_landmarks[22 - i][0]
        y_data[22 + i] = face_landmarks[22 - i][1]

    pts = np.vstack((x_data, y_data)).astype(np.int32).T
    cv2.fillConvexPoly(mask, pts, (255), 8, 0)

    result = cv2.bitwise_and(img, img, mask=mask)
    if display:
        cv2.imshow("mask", mask)
        # 根据mask，提取ROI区域
        cv2.imshow("result", result)
        cv2.waitKey(0)

    return result


def get_file_list(read_path):
    '''
    获取文件夹下图片的地址
    :param read_path:
    :return:
    '''
    path = read_path
    dirs = os.listdir(path)
    floder_len = len(dirs)
    file_name_list = []
    for i in range(floder_len):

        # 设置路径
        floder = dirs[i]
        floder_path = path + "/" + floder

        # 如果路径下是文件，那么就再次读取
        if os.path.isdir(floder_path):
            file_one = os.listdir(floder_path)
            file_len_one = len(file_one)
            for j in range(file_len_one):
                # 读取视频
                floder_path_one = floder_path + "/" + file_one[j]
                if os.path.isdir(floder_path_one):
                    file_two = os.listdir(floder_path_one)
                    file_len_two = len(file_two)
                    for k in range(file_len_two):
                        floder_path_two = floder_path_one + "/" + file_two[k]
                        if os.path.isdir(floder_path_two):
                            file_three = os.listdir(floder_path_two)
                            file_len_three = len(file_three)
                            for m in range(file_len_three):
                                floder_path_three = floder_path_two + "/" + file_three[m]
                                file_name_list.append(floder_path_three)
                        else:
                            file_name_list.append(floder_path_two)

                else:
                    file_name_list.append(floder_path_one)

        # 如果路径下，没有文件夹，直接是文件，就加入进来
        else:
            file_name_list.append(floder_path)

    return file_name_list


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    '求数据集的均值方差'
    '本质是读取一个epoch的数据进行测试,只不过把一个epoch的大小设置成了所有数据'
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]  # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def replace_string(path, index, new_one, split_str='/'):
    '''
    选择更换指定未知的字符串为某个新字符串
    :param path: 原始字符串
    : index 要修改的字符位置
    :new_one  更新的目标

    :return:
    '''
    path_split = path.split(split_str)
    path_split[index] = new_one
    path_new = split_str.join(path_split)
    return path_new


def seed_torch(seed=0):
    '''在使用模型的时候用于设置随机数'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def split_txt(txt_path, prop_list):
    '''
    将一个任务的验证或者测试集进行划分
    :param txt_path:
    :prop_list: 不同部分的占比,如[6,2,2]
    :return:
    '''

    txt_path_split = txt_path.split('/')
    txt_name = txt_path_split[-1]
    txt_index = txt_name.split('.')[0]
    txt_path_split.pop()
    save_dir = '/'.join(txt_path_split)

    txt_read = read_txt(txt_path)
    count = 0  # step=10,count 记录step的数量.
    index_min = 0
    while len(txt_read) / (count * 10 + 1) > 1:
        for prop in prop_list:
            print(index_min)
            prop_save_path = os.path.join(save_dir, txt_index + "_" + str(prop) + '.txt')
            read_part = txt_read[index_min + 10 * count:index_min + prop + 10 * count]
            index_min += prop
            with open(prop_save_path, 'a+') as f:
                for content in read_part:
                    f.write(content)
                    f.write('\n')
        count += 1
        index_min = 0


def add_txt(txt_path_1, txt_path_2, txt_path_new=None):
    '''
    合并两个txt
    :return:
    '''
    txt_path_1_read = read_txt(txt_path_1)
    if txt_path_new is None:
        txt_path_new = txt_path_2
        with open(txt_path_new, 'a+') as f:
            for content in txt_path_1_read:
                f.write(content)
                f.write('\n')
    else:
        txt_path_2_read = read_txt(txt_path_2)
        with open(txt_path_new, 'a+') as f:
            for content in txt_path_1_read:
                f.write(content)
                f.write('\n')
        with open(txt_path_new, 'a+') as f:
            for content in txt_path_2_read:
                f.write(content)
                f.write('\n')


def sort_via_assist(origin_list, assist_list):
    '''
    利用assist_list 帮助排序 origin_list
    :param sort_list:
    :param assist_list:
    :return:
    '''

    zipped = zip(origin_list, assist_list)
    sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]))
    result = zip(*sort_zipped)
    x_axis, y_axis = [list(x) for x in result]

    return x_axis, y_axis


def dataset_pair(multimodal_txt_path, unimodal_txt_path, pair_txt_path):
    '''
    将多模态数据集和单模态数据集,按照label 进行配对
    :return:
    '''
    multimodal_txt_read = read_txt(multimodal_txt_path)
    multimodal_label_list = []
    for index in range(len(multimodal_txt_read)):
        line_split = multimodal_txt_read[index].split(' ')
        if line_split[-1] == '':
            label = line_split[-2]
        else:
            label = line_split[-1]
        multimodal_label_list.append(int(label))
    multimodal_txt_read, multimodal_label_sort = sort_via_assist(multimodal_txt_read, multimodal_label_list)
    multimodal_living_number = sum(multimodal_label_sort)
    multimodal_spoofing_number = len(multimodal_label_sort) - multimodal_living_number
    print(multimodal_spoofing_number, multimodal_living_number)

    unimodal_txt_read = read_txt(unimodal_txt_path)
    unimodal_label_list = []
    for index in range(len(unimodal_txt_read)):
        line_split = unimodal_txt_read[index].split(' ')
        if line_split[-1] == ' ':
            label = line_split[-2]
        else:
            label = line_split[-1]
        # unimodal_txt_read[index] = line_split
        unimodal_label_list.append(int(label))
    unimodal_txt_read, unimodal_label_sort = sort_via_assist(unimodal_txt_read, unimodal_label_list)
    unimodal_living_number = sum(unimodal_label_sort)
    unimodal_spoofing_number = len(unimodal_label_sort) - unimodal_living_number
    print(unimodal_spoofing_number, unimodal_living_number)
    import time
    time.sleep(5)

    temp = []
    spoofing_prop = multimodal_spoofing_number // unimodal_spoofing_number
    spoofing_mod = np.mod(multimodal_spoofing_number, unimodal_spoofing_number)
    print(spoofing_prop, spoofing_mod)
    # time.sleep(5)
    unimodal_spoofing_file = unimodal_txt_read[0:unimodal_spoofing_number]
    # print(unimodal_spoofing_file)
    spoofing_tile = list(np.tile(unimodal_spoofing_file, spoofing_prop))
    temp = temp + spoofing_tile
    temp = temp + unimodal_txt_read[0:spoofing_mod]
    print(len(temp))

    living_prop = multimodal_living_number // unimodal_living_number
    living_mod = np.mod(multimodal_living_number, unimodal_living_number)
    unimodal_living_file = unimodal_txt_read[unimodal_spoofing_number:len(unimodal_txt_read)]
    living_tile = list(np.tile(unimodal_living_file, living_prop))
    temp = temp + living_tile
    temp = temp + unimodal_txt_read[unimodal_spoofing_number:unimodal_spoofing_number + living_mod]
    save_split = []
    for index in range(len(multimodal_label_sort)):
        multimodal_line = multimodal_txt_read[index]
        multimodal_line_split = multimodal_line.split(' ')

        unimodal_line = temp[index]
        unimodal_line_split = unimodal_line.split(' ')
        print(multimodal_line_split)
        print(unimodal_line_split)
        if multimodal_line_split[-1] == '':
            multimodal_line_split.pop()
        assert multimodal_line_split[-1] == unimodal_line_split[-1]

        save_split.append(multimodal_line_split)
        save_split[index].insert(0, unimodal_line_split[0])

    for line in save_split:
        save_csv(pair_txt_path, line)


def get_dataself_hist(arr):
    '''
    统计一个arr 不同数字出现的频率
    可以看做以本身为底 的直方图统计
    :param arr:
    :return:
    '''
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v

    return result


def transform_test(transform, img_path):
    '''
    用于测试设置的transform 是否有问题
    :param transform:
    :param img_path:
    :return:
    '''
    from PIL import Image

    img_pil = Image.open(img_path).convert(
        'RGB')
    img_r = transform(img_pil)
    img_b = transform(img_pil)
    img_r.show()
    img_b.show()


def save_csv(csv_path, data):
    '''
    以csv格式,增量保存数据,常用域log的保存

    :param csv_path: csv 文件地址
    :param data: 要保存数据,list和arr 都可以,但是只能是一维的
    :return:
    '''
    with open(csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)
    f.close()


def save_args(args, save_path):
    if os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()


def read_csv(csv_path):
    '''
    读取csv文件的内容,并且返回
    '''
    data_list = []
    csvFile = open(csv_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        '''item 是一个list,一个item 就是一行'''
        data_list.append(item)

    return data_list


def load_mat(mat_path):
    try:
        data = scio.loadmat(mat_path)
        data = data[list(data.keys())[-1]]
    except Exception as e:
        # data = h5py.File(mat_path)
        # data_key = data.keys()
        # print(data_key)
        # data = data[list(data_key)[0]]
        # print(data.shape)
        # print(dir(data))
        # data = data.value
        data = mat73.loadmat(mat_path)
        data = data[list(data.keys())[0]]
    return data


def read_txt(txt_path):
    '''
    读取txt 文件
    :param txt_path:
    :return: txt中每行的数据,结尾用'\n'
    '''

    f = open(txt_path)
    data = f.readlines()
    for index in range(len(data)):
        data[index] = data[index][:-1]
    return data


'''
数据和数据集的一些预处理操作,包括但不限于:从视频中获取图像,人脸检测,利用原始数据集,生成满足条件的数据集.
'''


def video_to_frames(pathIn='',
                    pathOut='',
                    extract_time_interval=-1,
                    only_output_video_info=False,
                    extract_time_points=None,
                    initial_extract_time=0,
                    end_extract_time=None,
                    output_prefix='frame',
                    jpg_quality=100,
                    isColor=True):
    '''
    pathIn：视频的路径，比如：F:\python_tutorials\test.mp4
    pathOut：设定提取的图片保存在哪个文件夹下，比如：F:\python_tutorials\frames1\。如果该文件夹不存在，函数将自动创建它
    only_output_video_info：如果为True，只输出视频信息（长度、帧数和帧率），不提取图片
    extract_time_points：提取的时间点，单位为秒，为元组数据，比如，(2, 3, 5)表示只提取视频第2秒， 第3秒，第5秒图片
    initial_extract_time：提取的起始时刻，单位为秒，默认为0（即从视频最开始提取）
    end_extract_time：提取的终止时刻，单位为秒，默认为None（即视频终点）
    extract_time_interval：提取的时间间隔，单位为秒，默认为-1（即输出时间范围内的所有帧）
    output_prefix：图片的前缀名，默认为frame，图片的名称将为frame_000001.jpg、frame_000002.jpg、frame_000003.jpg......
    jpg_quality：设置图片质量，范围为0到100，默认为100（质量最佳）
    isColor：如果为False，输出的将是黑白图片
    '''

    cap = cv2.VideoCapture(pathIn)  ##打开视频文件
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  ##视频的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  ##视频的帧率
    dur = n_frames / fps  ##视频的时间

    ##如果only_output_video_info=True, 只输出视频信息，不提取图片
    if only_output_video_info:
        print('only output the video information (without extract frames)::::::')
        print("Duration of the video: {} seconds".format(dur))
        print("Number of frames: {}".format(n_frames))
        print("Frames per second (FPS): {}".format(fps))

        ##提取特定时间点图片
    elif extract_time_points is not None:
        if max(extract_time_points) > dur:  ##判断时间点是否符合要求
            raise NameError('the max time point is larger than the video duration....')
        try:
            os.mkdir(pathOut)
        except OSError:
            pass
        success = True
        count = 0
        while success and count < len(extract_time_points):
            cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * extract_time_points[count]))
            success, image = cap.read()
            if success:
                if not isColor:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ##转化为黑白图片
                print('Write a new frame: {}, {}th'.format(success, count + 1))
                cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                count = count + 1

    else:
        ##判断起始时间、终止时间参数是否符合要求
        if initial_extract_time > dur:
            raise NameError('initial extract time is larger than the video duration....')
        if end_extract_time is not None:
            if end_extract_time > dur:
                raise NameError('end extract time is larger than the video duration....')
            if initial_extract_time > end_extract_time:
                raise NameError('end extract time is less than the initial extract time....')

        ##时间范围内的每帧图片都输出
        if extract_time_interval == -1:
            if initial_extract_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time))
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) * fps + 1
                success = True
                count = 0
                while success and count < N:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1

        ##判断提取时间间隔设置是否符合要求
        elif extract_time_interval > 0 and extract_time_interval < 1 / fps:
            raise NameError('extract_time_interval is less than the frame time interval....')
        elif extract_time_interval > (n_frames / fps):
            raise NameError('extract_time_interval is larger than the duration of the video....')

        ##时间范围内每隔一段时间输出一张图片
        else:
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) / extract_time_interval + 1
                success = True
                count = 0
                while success and count < N:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1


def frame_to_face(frame_dir, face_dir, model_name, normal_size=None, save_mode='.jpg'):
    '''
    检测人脸，并保存
    :param frame_dir: 原始帧保留的文件夹
    :param face_dir: 人脸保留的位置
    :model_name: 人脸检测的模型
    :return:
    '''
    face_detector = FaceDection(model_name=model_name)

    for root, dirs, files in os.walk(frame_dir):

        # 当子目录为空的时候，root就是不包含子文件夹的文件夹
        if dirs == []:

            for file_name in files:
                file_path = os.path.join(root, file_name)
                img = cv2.imread(file_path)
                if img is None:
                    print("if img is None:")
                    continue

                # 人脸检测
                if normal_size is not None:
                    img_normal = cv2.resize(img, normal_size)
                else:
                    img_normal = img
                face_img = face_detector.face_detect(img_normal)
                if face_img is None:
                    print("if face_img is None:")
                    continue

                # 获取存储路径
                frame_dir_split = frame_dir.split('/')
                file_path_split = file_path.split('/')
                file_path_split.pop()  # 去掉文件名
                sub_split = [item for item in file_path_split if item not in frame_dir_split]
                save_dir = face_dir
                for item in sub_split:
                    save_dir = os.path.join(save_dir, item)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 存储
                save_path = os.path.join(save_dir, file_name.split('.')[0] + save_mode)
                cv2.imwrite(save_path, face_img)


def analysis(true_right, true_wrong, false_right, false_wrong):
    false_all = false_wrong + false_right
    true_all = true_wrong + true_right
    true_class = true_right + false_wrong
    false_class = true_wrong + false_right
    all = true_right + true_wrong + false_wrong + false_right

    accuracy = (true_right + false_right) / all
    APCER = false_wrong / false_all
    BPCER = true_wrong / true_all
    ACER = (APCER + BPCER) / 2
    FAR = false_wrong / true_class
    FRR = true_wrong / false_class

    print("accuracy", accuracy, "APCER", APCER, "BPCER", BPCER, "ACER", ACER, "FAR", FAR, "FRR", FRR)


def img_preview(img_dir):
    file_list = get_file_list(img_dir)
    for file_path in file_list:
        img = cv2.imread(file_path)
        if img is None:
            continue
        img_shape = img.shape
        print("img_shape", img_shape)
        cv2.namedWindow("img_show", 0)
        cv2.imshow("img_show", img)
        cv2.waitKey(0)


def x_to_jpg(x_path):
    img_path_list = get_file_list(x_path)
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        save_path = replace_string(img_path, -1, 'jpg', split_str='.')
        print(save_path)
        cv2.imwrite(save_path, img)


def process_hdf5():
    f = h5py.File('/home/shicaiwei/data/liveness_data/HQ-WMCA/1_01_0092_0000_00_00_000-fad05fcd.hdf5', 'r')
    print(f.filename, ":")
    print([key for key in f.keys()], "\n")
    d = f["Frame_2"]
    for k in d.keys():
        print(k)
    d = d['array']
    for i in range(d.shape[0]):
        sample = d[i, :, :]
        cv2.imshow("sample", sample)
        cv2.waitKey(0)
    print(d.shape)
    print(d.value)
    print(1)


def generate_relative_path(dataset_root):
    '''
    为单个数据集生成相对路径
    :param dataset_root:
    :return:
    '''
    file_path_list = get_file_list(dataset_root)
    dataset_root_split = dataset_root.split('/')
    dataset_root_split_set = set(dataset_root_split)
    for path in file_path_list:
        path_split = path.split('/')
        path_split_set = set(path_split)
        relative_path_set = path_split_set - dataset_root_split_set
        relative_path_list = list(relative_path_set)
        relative_path = '/'.join(relative_path_list)
        return relative_path


def get_prediction(outputs_full, labels_full):
    '''
    目的是为了获取最优的划分边界,但是只适合二分类?
    输入是所有输出预测输出的logits的集成和 label.()
    :param outputs_full: sample x class_num
    label_full, sample x 1
    :return:
    '''
    predict_prob = torch.softmax(outputs_full.data, 1)
    predict_prob = predict_prob[:, 1]
    print(predict_prob.shape)
    fpr_test, tpr_test, threshold_test = roc_curve(labels_full.cpu().numpy(), predict_prob.cpu().numpy(), pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)
    print(err_test, best_test_threshold)

    labels_predicted = torch.softmax(outputs_full.data, dim=1)
    labels_predicted = labels_predicted[:, 1]
    labels_predicted[labels_predicted > best_test_threshold] = 1
    labels_predicted[labels_predicted < 1] = 0


def cam_analysii(model, img_path):
    model = resnet18(pretrained=True).eval()
    cam_extractor = SmoothGradCAMpp(model)
    # Get your input
    img = cv2.imread(img_path)
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    import matplotlib.pyplot as plt
    # Visualize the raw CAM
    plt.imshow(activation_map[0].numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def model_infer_func(model, val_loader):
    '''
    需要单独写这个函数是为了兼容不同模型的输入可能不同,推理过程可能不同,难以自适应
    可以直接复制val/test的代码就可以.
    :param model:
    :param datalodaer:
    :return:
    '''
    target_list = []
    output_list = []
    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output, patch, _ = model(input, is_student=False)
            patch_score = patch
            target_list.append(target)
            output_list.append(output)

        output_list = torch.cat(output, dim=0)
        target_list = torch.cat(output, dim=0)
        output_arr = np.array(output_list.cpu())
        target_arr = np.array(target_list.cpu())
        output_arr = np.transpose(output_arr)
        target_arr = np.transpose(target_arr)
        return output_arr, target_arr


def get_model_infer_result(model_origin, model_enhance, val_dataloader, model_infer_func):
    origin_out, origin_label = model_infer_func(model_origin, val_dataloader)
    enhance_out, enhance_label = model_infer_func(model_enhance, val_dataloader)

    with open('origin_out.csv', 'a+', newline='') as f:
        for i in range(origin_out.shape[0]):
            writer = csv.writer(f)
            row = origin_out[i, :]
            row = list(row.cpu().detach().numpy())
            writer.writerow(row)

    with open('enhance_out.csv', 'a+', newline='') as f:
        for i in range(enhance_out.shape[0]):
            writer = csv.writer(f)
            row = enhance_out[i, :]
            row = list(row.cpu().detach().numpy())
            writer.writerow(row)

    with open('label.csv', 'a+', newline='') as f:
        for i in range(origin_label.shape[0]):
            writer = csv.writer(f)
            row = origin_label[i, :]
            row = list(row.cpu().detach().numpy())
            writer.writerow(row)


def predicction_analysis():
    '''
    对分类结果进行分析的代码片段,为了获取每个模型的预测结果,和预测结果的差异,并分析差异成分.
    :return:
    '''

    # 获取预测结果
    target_list = []
    target_numpy = []
    patch_score = torch.tensor([1.8, 2.2])  # 预测的logit 输出
    with open('model_origin_prediction.csv', 'a+', newline='') as f:
        for i in range(patch_score.shape[0]):
            writer = csv.writer(f)
            for j in range(patch_score.shape[2]):
                row = patch_score[i, :, j]
                row = list(row.cpu().detach().numpy())
                writer.writerow(row)


if __name__ == '__main__':
    a = [1, 2, 4, 3]
    b = ['a', 'c', 'd', 'e']
    c, d = sort_via_assist(b, a)
    print(c, d)
