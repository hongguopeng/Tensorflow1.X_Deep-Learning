import numpy as np
import scipy.misc as sc
import glob
import os
import matplotlib.pyplot as plt

# 影像下載網址:https://www.kaggle.com/jessicali9530/celeba-dataset
img_path = os.path.join(os.getcwd() , 'original')
resize_path = os.path.join(os.getcwd() , 'resize')
mosaic_path = os.path.join(os.getcwd() , 'mosaic')
img_list = glob.glob(img_path + '/*.jpg')

height = 256
width  = 256
images_resize = np.zeros([len(img_list) , height , width , 3])
images_mosaic = np.zeros([len(img_list) , height , width , 3])

for i , img_file in enumerate(img_list):
    img = sc.imread(img_file)
    img_resize = sc.imresize(img , [height , width , 3] ,
                             interp = 'bilinear' , mode = 'RGB')
    
    images_resize[i , : , : , :] = img_resize
    sc.imsave(os.path.join(resize_path , os.path.basename(img_file)) , img_resize)

    w = img_resize.shape[0]
    img_mosaic = img_resize.copy()
    # 將影像加上馬賽克，增加a可以增加馬賽克單元格的尺寸
    a = 6
    for j in range(int(1 / 4 * w), int(3 / 4 * w), a):
        for k in range(int(1 / 4 * w), int(3 / 4 * w), a):
            img_mosaic[j:j + a, k:k + a] = img_mosaic[j + (a // 2)][k + (a // 2)]
 
    images_mosaic[i , : , : , :] = img_mosaic
    sc.imsave(os.path.join(mosaic_path , os.path.basename(img_file)) , img_mosaic)
 
np.save('images_resize' , images_resize)
np.save('images_mosaic' , images_mosaic)

