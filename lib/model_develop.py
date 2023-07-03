'''相比于model_develop_utils.py, 这部分的代码更有特异性,针对不同领域做了调整'''

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
import time

import os
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F

from lib.model_develop_utils import GradualWarmupScheduler, calc_accuracy


def calc_accuracy_multi(model, loader, verbose=False, hter=False):
    '''
    针对多模态任务,修改了dataloader
    :param model:
    :param loader:
    :param verbose:
    :param hter:
    :return:
    '''
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                             batch_sample['image_depth'], batch_sample[
                                                 'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_rgb, img_ir, img_depth)

            # 如果有模型有多个返回
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:
            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            ACER = (APCER + NPCER) / 2
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            ACER = float("%.6f" % ACER)
            accuracy = float("%.6f" % accuracy)

        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 1, 1, 1, 1, 1, 1, 1]
        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER]
    else:
        return [accuracy]


def calc_accuracy_kd(model, loader, args, verbose=False, hter=False):
    '''
    针对跨模态蒸馏学习修改了dataloader部分,加载多个模态的数据,选择某个目标模态进行测试
    :param model:
    :param loader:
    :param args:
    :param verbose:
    :param hter:
    :return:
    '''

    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                             batch_sample['image_depth'], batch_sample[
                                                 'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        if args.student_data == 'multi':
            with torch.no_grad():
                outputs_batch = model(img_rgb, img_depth, img_ir)
                # 如果有多个返回值只取第一个
                if isinstance(outputs_batch, tuple):
                    outputs_batch = outputs_batch[0]
        else:
            if args.student_data == 'multi_rgb':
                test_data = img_rgb
            elif args.student_data == 'multi_depth':
                test_data = img_depth
            elif args.student_data == 'multi_ir':
                test_data = img_ir
            else:
                test_data = img_rgb
                print('test_error')
            with torch.no_grad():
                outputs_batch = model(test_data)

                # 如果有多个返回值只取第一个
                if isinstance(outputs_batch, tuple):
                    outputs_batch = outputs_batch[0]
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def train_base_multi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数,相比于train_base ,增加修改了dataloader数据读取和模型输入
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    hter_best = 1
    acer_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                                 batch_sample['image_depth'], batch_sample[
                                                     'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            output = model(img_rgb, img_ir, img_depth)

            loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, loader=test_loader)
        print(result_test)
        accuracy_test = result_test[0]
        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 0:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)

# def train_pixel_supervise(model, cost, optimizer, train_loader, test_loader, args):
#     '''
#     适用于回归,分割等图像监督的训练任务,相比于train base
#     :param model:
#     :param cost:
#     :param optimizer:
#     :param train_loader:
#     :param test_loader:
#     :param args:
#     :return:
#     '''
#
#     print(args)
#
#     # Initialize and open timer
#     start = time.time()
#
#     if not os.path.exists(args.model_root):
#         os.makedirs(args.model_root)
#     if not os.path.exists(args.log_root):
#         os.makedirs(args.log_root)
#
#     models_dir = args.model_root + '/' + args.name + '.pt'
#     log_dir = args.log_root + '/' + args.name + '.csv'
#
#     # save args
#     with open(log_dir, 'a+', newline='') as f:
#         my_writer = csv.writer(f)
#         args_dict = vars(args)
#         for key, value in args_dict.items():
#             my_writer.writerow([key, value])
#         f.close()
#
#     # Cosine learning rate decay
#     if args.lr_decrease:
#         print("lrcos is using")
#         cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=0)
#
#         if args.lr_warmup:
#             scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
#                                                       after_scheduler=cos_scheduler)
#
#     # Training initialization
#     epoch_num = args.train_epoch
#     log_interval = args.log_interval
#     save_interval = args.save_interval
#     batch_num = 0
#     train_loss = 0
#     epoch = 0
#     loss_best = 1e4
#     log_list = []  # log need to save
#
#     if args.retrain:
#         if not os.path.exists(models_dir):
#             print("no trained model")
#         else:
#             state_read = torch.load(models_dir)
#             model.load_state_dict(state_read['model_state'])
#             optimizer.load_state_dict(state_read['optim_state'])
#             epoch = state_read['Epoch']
#             print("retaining")
#
#     # Train
#     while epoch < epoch_num:
#         for batch_idx, (data, target) in enumerate(
#                 tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):
#
#             batch_num += 1
#
#             target = torch.unsqueeze(target, dim=1)
#
#             if torch.cuda.is_available():
#                 data, target = data.cuda(), target.cuda()
#
#             if args.mixup:
#                 mixup_alpha = args.mixup_alpha
#                 inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)
#
#             optimizer.zero_grad()
#
#             output = model(data)
#
#             if args.mixup:
#                 loss = mixup_criterion(cost, output, labels_a, labels_b, lam)
#             else:
#                 loss = cost(output, target)
#
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
#             #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             #         epoch, batch_idx * len(data), len(train_loader.dataset),
#             #                100. * batch_idx / len(train_loader), loss.item()))
#
#         # testing
#         result_test = calc_accuracy_pixel(model, loader=test_loader)
#         test_loss = result_test
#         if test_loss < loss_best:
#             loss_best = train_loss / len(train_loader)
#             save_path = args.model_root + args.name + '.pth'
#             torch.save(model.state_dict(), save_path)
#         log_list.append(test_loss)
#
#         print(
#             "Epoch {}, loss={:.5f}".format(epoch,
#                                            train_loss / len(train_loader),
#                                            ))
#         train_loss = 0
#         if args.lr_decrease:
#             if args.lr_warmup:
#                 scheduler_warmup.step(epoch=epoch)
#             else:
#                 cos_scheduler.step(epoch=epoch)
#         if epoch < 20:
#             print(epoch, optimizer.param_groups[0]['lr'])
#
#         # save model and para
#         if epoch % save_interval == 0:
#             train_state = {
#                 "Epoch": epoch,
#                 "model_state": model.state_dict(),
#                 "optim_state": optimizer.state_dict(),
#                 "args": args
#             }
#             models_dir = args.model_root + '/' + args.name + '.pt'
#             torch.save(train_state, models_dir)
#
#         #  save log
#         with open(log_dir, 'a+', newline='') as f:
#             # 训练结果
#             my_writer = csv.writer(f)
#             my_writer.writerow(log_list)
#             log_list = []
#         epoch = epoch + 1
#     train_duration_sec = int(time.time() - start)
#     print("training is end", train_duration_sec)
