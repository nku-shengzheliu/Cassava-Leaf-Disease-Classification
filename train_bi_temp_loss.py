# train.py
# !/usr/bin/env	python3

import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data.distributed
from torch.autograd import Variable
from torch.utils.data import WeightedRandomSampler

from conf import settings
from utils import Single_Dataset, WarmUpLR
from torch.nn import init
from models.efficientNet.model import EfficientNet
from trick import label_smoothing, cutmix, bi_tempered_loss
import numpy as np

cnt_print = -1
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


def train(epoch):
    global cnt_print

    correct = 0.0
    sampleNum = 0
    torch.cuda.empty_cache()
    net.train()
    if global_rank == 0:
        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    for batch_index, (images, labels) in enumerate(training_loader):

        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()
        optimizer.zero_grad()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        sampleNum += outputs.size()[0]

        loss.backward()
        optimizer.step()

        if global_rank == 0:
            cnt_print = cnt_print + 1
            if cnt_print%50 == 0:
                cnt_print = 0
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * args.b + len(images),
                    total_samples=len(training_loader.dataset)
                ))
    print("Train Accuracy :{:0.4f}".format(correct.float() / sampleNum))


def eval_training(epoch):
    torch.cuda.empty_cache()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = Variable(images)
            labels = Variable(labels)

            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)

            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
        if global_rank == 0:
            logging.info('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
                test_loss / len(test_loader.dataset),
                correct.float() / len(test_loader.dataset)
            ))
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
                test_loss / len(test_loader.dataset),
                correct.float() / len(test_loader.dataset)
            ))
            print()


    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = settings.GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='efficientnet-b4', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.02, help='initial learning rate')
    parser.add_argument('-workers', type=int, default=8, help='number of Dataloader workers')
    parser.add_argument('-cutmix_beta', type=float, default=0.3)
    parser.add_argument('-cutmix_prob', type=float, default=0.3)
    args = parser.parse_args()

    global_rank = 0
    world_size = 1
    net =EfficientNet.from_pretrained(args.net,in_channels=3,num_classes=5)
    #net._conv_stem.apply(weights_init)
    net._fc.apply(weights_init)

    net = torch.nn.DataParallel(net,device_ids = (0,1,)).cuda()

    # # setting training_loader
    train_transform = transforms.Compose([
                                          transforms.RandomCrop((480, 480)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.42984136, 0.49624753, 0.3129598), std=(0.21417203, 0.21910103, 0.19542212))
                                          ])

    train_dataset = Single_Dataset(image_root=r'./cassava-leaf-disease-classification/train_images_resize',
                                   txt_path= './train.txt',
                                   is_train=True,
                                   transform = train_transform
                                   )
    prob_list = open('./prob.txt').readlines()
    prob_list = list(map(lambda x:float(x.strip('\n')),prob_list))

    training_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.b // world_size,
                                                  shuffle=True,
                                                  #sampler=train_sampler,
                                                  num_workers=args.workers // world_size,
                                                  pin_memory=True
                                                  )
    # setting test_loader
    test_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.42984136, 0.49624753, 0.3129598), std=(0.21417203, 0.21910103, 0.19542212))
                                         ])
    test_dataset = Single_Dataset(
                                    image_root=r'./cassava-leaf-disease-classification/train_images_resize',
                                    txt_path='./val.txt',
                                    is_train=False,
                                    transform=test_transform
                                  )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.b ,
                                              shuffle=False,
                                              num_workers=args.workers
                                              )

    #loss_function = nn.CrossEntropyLoss()
    # loss_function = label_smoothing.LabelSmoothSoftmaxCEV1(lb_smooth=0.1)
    loss_function = bi_tempered_loss.bi_tempered_logistic_loss()

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
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=settings.GAMMA)  # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use Logging
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if global_rank == 0:
        logging.basicConfig(
            filename=os.path.join(settings.LOG_DIR, "train.log"),
            filemode='a+',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    if global_rank == 0:
        logging.info('Network weights save to {}'.format(checkpoint_path))
        logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                     format(settings.EPOCH, args.b // world_size, len(train_dataset), len(test_dataset)))  # len(validate_dataset)))
        logging.info('')
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):

        if epoch > args.warm:
            train_scheduler.step(epoch)
        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if best_acc < acc and global_rank == 0:
            torch.save(net.module.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH and global_rank == 0:
            torch.save(net.module.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    # writer.close()
