""" configurations for this project
"""
from datetime import datetime

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
GAMMA = 0.1

#total training epoches
EPOCH = 25
MILESTONES = [10, 18]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()[:10]

#log dir
LOG_DIR = 'logs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 1

# GPU
GPU = '0,1'
