import matplotlib.pyplot as plt
from astropy.io import fits
import os
import warnings
import numpy as np
from PIL import Image
import numpy as np

warnings.filterwarnings('ignore')

type1  = ['magnetogram','continuum']


for type_1 in type1:

    path = '../../dataset/'+'test/'+type_1
    save = '../../dataset/'+'test/'+type_1+'_NPY'
    if not os.path.exists(save):
        os.mkdir(save)
    # save_test = './img/'
    # temp = fits.open("hmi.sharp_720s.10.20100503_000000_TAI.continuum.fits")
    list = os.listdir(path)
    for i in range(0, len(list)):
        # print(list[i])
        f_path = os.path.join(path, list[i])
        save_path = os.path.join(save, list[i])
        save_path = save_path[:-5]
        save_path = save_path + '.npy'
        # print(save_path)

        temp = fits.open(f_path)
        # print(temp.info())
        temp.verify('fix')
        npy = temp[1].data
        npy[np.isnan(npy)] = 0
        np.save(save_path,npy)
