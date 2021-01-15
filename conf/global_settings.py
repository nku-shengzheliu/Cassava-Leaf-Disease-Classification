""" configurations for this project

"""
import os
from datetime import datetime


#mean and std of dataset
# TRAIN_MEAN_C = (0.80664591,) #continuum观测图像的均值
# TRAIN_STD_C = (0.07190562,) #continuum观测图像的标准差
# TRAIN_MEAN_M = (0.50427939,) #magnetogram观测图像的均值
# TRAIN_STD_M = (0.04170241,) #magnetogram观测图像的标准差
#
# TEST_MEAN_C = (0.81050262,)
# TEST_STD_C = (0.07015978,)
# TEST_MEAN_M = (0.49208785,)
# TEST_STD_M =(0.03897226,)
TRAIN_MEAN_C = (25603.861,) #continuum观测图像的均值
TRAIN_STD_C = (26455.03,) #continuum观测图像的标准差
TRAIN_MEAN_M = (25618.357,) #magnetogram观测图像的均值
TRAIN_STD_M = (26455.96,) #magnetogram观测图像的标准差

TEST_MEAN_C = (25603.861,)
TEST_STD_C = (26455.03,)
TEST_MEAN_M = (25618.357,)
TEST_STD_M =(26455.96,)

TRAIN_MEAN = (0.65546265, 0.65546265, 0.65546265)
TRAIN_STD = (0.05680402, 0.05680402, 0.05680402)

TEST_MEAN = (0.65129524,0.65129524,0.65129524)
TEST_STD = (0.05456602,0.05456602,0.05456602)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
GAMMA = 0.1

#total training epoches
EPOCH = 100
MILESTONES = [15,30,60,75]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()[:10]

#log dir
LOG_DIR = 'logs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

# GPU
GPU = '0,1'






