import numpy as np
import scipy.misc as sc
import glob
import os
import matplotlib.pyplot as plt

# 圖片下載網址
# Training Data ⇒ https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip
# Training Ground Truth ⇒ https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip

img_path = os.path.join(os.getcwd() , 'Training_Data')
mask_path = os.path.join(os.getcwd() , 'Training_GroundTruth')
img_list = glob.glob(img_path + '/*.jpg')
mask_list = glob.glob(mask_path + '/*.png')

# 對每張圖片做resize
height = 256
width  = 256
images = np.zeros([len(img_list) , height , width , 3])
masks = np.zeros([len(img_list) , height , width , 1])
for idx in range(0 , len(img_list)):
    img = sc.imread(img_list[idx])
    img = sc.imresize(img , [height , width , 3] ,
                      interp = 'bilinear' , mode = 'RGB')
    images[idx , : , : , :] = img

    mask = sc.imread(mask_list[idx])
    mask = sc.imresize(mask , [height , width] , interp = 'bilinear')
    masks[idx , : , : , 0] = mask

    print('processing : {}'.format(idx))

images = images[:1000]
masks = masks[:1000]
np.save('images' , images)
np.save('masks ' , masks )
