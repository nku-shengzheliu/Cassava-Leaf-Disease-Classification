import matplotlib.pyplot as plt
from astropy.io import fits
import os
import warnings
import numpy as np
from PIL import Image
import numpy as np

warnings.filterwarnings('ignore')

type1  = ['magnetogram','continuum']
type2 = ['alpha','beta','betax']

for type_1 in type1:
    for type_2 in type2:
        path = '../../dataset/'+type_1+'/'+ type_2
        save = '../../dataset/'+type_1+'_NPY'+'/'+type_2
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
            # mmax = img.max()
            # mmin = img.min()
            # print(img.shape)
            # 保存fits图像
            # im = Image.fromarray(np.uint8((img-mmin)/(mmax-mmin)*255))
            # im = im.convert('RGB')
            # print(len(im.split()))
            # print(type(im))
            # im.save(save_path)