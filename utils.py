"""
dataset function
"""

import sys
import numpy
import os
from astropy.io import fits
import warnings
from PIL import Image
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


class Single_Dataset(Dataset):
    def __init__(self,image_root, txt_path,is_train , transform = None):
        super(Single_Dataset,self).__init__()
        self.image_root = image_root
        self.txt_path = txt_path
        self.is_train = is_train
        self.transform = transform
        self.image_list = []
        self.label_list = []

        lines = open(self.txt_path,'r').readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            self.image_list.append(line.split('   ')[0])
            self.label_list.append(int(line.split('   ')[1]))


        self.seq = iaa.SomeOf((3, 11), {
            # self.seq = iaa.SomeOf((0, 5), {
            # iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            # iaa.Crop(percent=(0, 0.1)),
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # 先将图片从RGB变换到HSV,然后将H值增加10,然后再变换回RGB
            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                               children=iaa.WithChannels(2, iaa.Add((10, 50)))),
            iaa.AverageBlur(k=((2, 5), (1, 3))),
            iaa.SimplexNoiseAlpha(
                first=iaa.EdgeDetect((0.0, 0.2)),
                second=iaa.ContrastNormalization((0.5, 2.0)),
                per_channel=True
            ),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.ImpulseNoise(p=0.02),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.PerspectiveTransform(scale=0.06),
            # # 图像扭曲
            # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-8, 8)
            )
        }, random_order=True)

    def __getitem__(self, index):
        image_path  = os.path.join(self.image_root,self.image_list[index])
        label = self.label_list[index]
        img = Image.open(image_path).convert('RGB')
        if self.is_train:
            img = self.seq.augment_image(np.array(img))
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img,label

    def __len__(self):
        return len(self.image_list)


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
