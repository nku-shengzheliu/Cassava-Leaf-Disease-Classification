# train.py
# !/usr/bin/env	python3

import os
import argparse
import logging
import random
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data.distributed
from torch.autograd import Variable
from torch.utils.data import WeightedRandomSampler

from conf import settings_cv
from dataset import CLDC_Dataset, WarmUpLR
from torch.nn import init
from models.efficientNet.model import EfficientNet
from trick import label_smoothing, cutmix, bi_tempered_loss
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        if m.bias is not None:
            init.constant(m.bias, 0)

def get_model():
    net = EfficientNet.from_pretrained(args.net, in_channels=3, num_classes=5)
    # net._conv_stem.apply(weights_init)
    net._fc.apply(weights_init)
    net = torch.nn.DataParallel(net, device_ids=(0, 1,)).cuda()
    return net

def get_dataloader(data_train, data_valid):
    train_transform = transforms.Compose([
        transforms.RandomCrop((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.42984136, 0.49624753, 0.3129598), std=(0.21417203, 0.21910103, 0.19542212))
    ])
    train_dataset = CLDC_Dataset(
        image_root=r'/home/ubuntu6/lsz/dataset/cassava-leaf-disease-classification/train_images_resize',
        data_list= data_train,
        is_train=True,
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.b,
                                                  shuffle=True,
                                                  # sampler=train_sampler,
                                                  num_workers=args.workers,
                                                  pin_memory=True
                                                  )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.42984136, 0.49624753, 0.3129598), std=(0.21417203, 0.21910103, 0.19542212))
    ])
    val_dataset = CLDC_Dataset(
        image_root=r'/home/ubuntu6/lsz/dataset/cassava-leaf-disease-classification/train_images_resize',
        data_list= data_valid,
        is_train=False,
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.b,
                                              shuffle=False,
                                              num_workers=args.workers
                                              )
    return train_loader, val_loader

def get_kfold_data(k, i, train_list):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(train_list) // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        data_valid = train_list[val_start:val_end]
        data_train = train_list[0:val_start] + train_list[val_end:]
    else:  # 若是最后一折交叉验证
        data_valid = train_list[val_start:]  # 若不能整除，将多的case放在最后一折里
        data_train = train_list[0:val_start]

    return data_train, data_valid

def traink(net, data_train, data_valid, args, current_k):
    # make dataloader
    train_loader, val_loader = get_dataloader(data_train, data_valid)

    # fine-tuning
    params = []
    learning_rate = args.lr
    new_param_names = ['fc', 'margin', 'attentions', 'bap']
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if any(i in key for i in new_param_names) and "layer" not in key:
                if 'bn' not in key:
                    params += [{'params': [value], 'lr': learning_rate, 'weight_decay': 5e-4}]
                else:
                    params += [{'params': [value], 'lr': learning_rate}]
            else:
                if 'bn' not in key:
                    params += [{'params': [value], 'lr': learning_rate * 0.2, 'weight_decay': 5e-4}]
                else:
                    params += [{'params': [value], 'lr': learning_rate * 0.2}]

    # loss and optimizer
    loss_function = label_smoothing.LabelSmoothSoftmaxCEV1(lb_smooth=0.1)
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings_cv.MILESTONES,
                                                     gamma=settings_cv.GAMMA)  # learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings_cv.CHECKPOINT_PATH, args.net, settings_cv.TIME_NOW, str(current_k))

    # use Logging
    if not os.path.exists(settings_cv.LOG_DIR):
        os.mkdir(settings_cv.LOG_DIR)
    logging.basicConfig(
        filename=os.path.join(settings_cv.LOG_DIR, args.logname),
        filemode='a+',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    logging.info('Network weights save to {}'.format(checkpoint_path))
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(settings_cv.EPOCH, args.b, len(data_train),
                        len(settings_cv)))  # len(validate_dataset)))
    logging.info('')

    losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(settings_cv.EPOCH):
        correct = 0.0  # 记录正确的个数，每个epoch训练完成之后打印accuracy
        sampleNum = 0
        net.train()
        # Freeze BN
        freeze_bn = True  # Freezing Mean/Var of BatchNorm2D
        freeze_bn_affine = True  # Freezing Weight/Bias of BatchNorm2D
        if freeze_bn:
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    print(m)
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if epoch > args.warm:
            train_scheduler.step(epoch)

        for batch_index, (images, labels) in enumerate(train_loader):
            if epoch <= args.warm:
                warmup_scheduler.step()
            images = Variable(images)
            labels = Variable(labels)
            labels = labels.cuda()
            images = images.cuda()
            optimizer.zero_grad()
            outputs = net(images)
            # 计算损失函数
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # 计算正确率
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
            sampleNum += outputs.size()[0]

            if (i + 1) % 10 == 0:
                # 每10个batches打印一次loss
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f  LR: %.6f' % (epoch + 1, settings_cv.EPOCH,
                                                                    i + 1, iter_per_epoch // args.b,
                                                                    loss.item(),optimizer.param_groups[0]['lr']))
        accuracy = correct.float() / sampleNum
        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.4f}%)'.format(
            epoch + 1, loss.item(), correct, sampleNum, accuracy))
        train_acc.append(accuracy)

        # 验证
        net.eval()
        val_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = Variable(images)
                labels = Variable(labels)
                images = images.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)  # batch average loss
                val_loss += loss.item()  # sum up batch loss
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum()

        val_losses.append(val_loss / len(val_loader.dataset))
        accuracy = 100. * correct / len(val_loader.dataset)
        logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset), accuracy))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset), accuracy))
        val_acc.append(accuracy)
        # start to save best performance model after learning rate decay to 0.01
        if best_acc < accuracy:
            torch.save(net.module.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = accuracy
            continue
        if not epoch % settings_cv.SAVE_EPOCH:
            torch.save(net.module.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    return losses, val_losses, train_acc, val_acc


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = settings_cv.GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='efficientnet-b4', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-k', type=int, default=5, help='k-fold cross validation')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.02, help='initial learning rate')
    parser.add_argument('-workers', type=int, default=8, help='number of Dataloader workers')
    parser.add_argument('-logname', type=str, default="train_btloss_cv.log", help='log name')
    parser.add_argument('-cutmix_beta', type=float, default=0.3)
    parser.add_argument('-cutmix_prob', type=float, default=0.3)
    args = parser.parse_args()

    # 加载全部训练文件列表
    train_list = []
    csv_file = csv.reader(open('train.csv','r'))
    for train_data in csv_file:
        train_list.append(train_data)
    random.shuffle(train_list)  # 打乱数据列表

    # k折交叉验证
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    for i in range(args.k):
        print('*' * 25, 'The', i + 1, 'fold', '*' * 25)
        data_train, data_valid = get_kfold_data(args.k, i, train_list)  # 获取k折交叉验证的训练和验证数据
        net = get_model()  # 实例化模型（某已经定义好的模型）

        # 每份数据进行训练
        train_loss, val_loss, train_acc, val_acc = traink(net, data_train, data_valid, args, i)

        print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[-1], train_acc[-1]))
        print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss[-1], val_acc[-1]))

        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += val_acc[-1]

    print('\n', '#' * 10, 'The final k-fold cross validation results', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / args.k, train_acc_sum / args.k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / args.k, valid_acc_sum / args.k))
