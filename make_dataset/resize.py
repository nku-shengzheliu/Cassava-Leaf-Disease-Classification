import cv2
import os
from tqdm import tqdm

image_path = "../cassava-leaf-disease-classification/train_images"
save_path = "../cassava-leaf-disease-classification/train_images_resize/"
image_list = os.listdir(image_path)

for imagename in tqdm(image_list):
    # print(imagename)
    image = cv2.imread(image_path + '/' + imagename)
    x, y = image.shape[0:2]
    image = cv2.resize(image, (512, 512))
    #image = cv2.resize(image,(H/2, W/2, 3))
    cv2.imwrite(save_path+imagename,image)
    # break
