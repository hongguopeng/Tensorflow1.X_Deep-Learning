import numpy as np
import scipy.misc as sc
import cv2
import glob
import os
import matplotlib.pyplot as plt

# 影像下載網址:https://www.kaggle.com/jessicali9530/celeba-dataset
img_path = os.path.join(os.getcwd() , 'original')
resize_path = os.path.join(os.getcwd() , 'resize')
blur_path = os.path.join(os.getcwd() , 'blur')
img_list = glob.glob(img_path + '/*.jpg')

height = 256
width  = 256
# 可以增加kernel_size與sigma，增加影像模糊程度
kernel_size = (11 , 11)
sigma = 2
images_resize = np.zeros([len(img_list) , height , width , 3])
images_blur = np.zeros([len(img_list) , height , width , 3])

for i , img_file in enumerate(img_list):
    img = sc.imread(img_file)
    img_resize = sc.imresize(img , [height , width , 3] ,
                             interp = 'bilinear' , mode = 'RGB')
    
    images_resize[i , : , : , :] = img_resize
    sc.imsave(os.path.join(resize_path , os.path.basename(img_file)) , img_resize)

    w = img_resize.shape[0]
    img_blur = img_resize.copy()
    img_blur = cv2.GaussianBlur(img_blur , kernel_size , sigma)
    images_blur[i , : , : , :] = img_blur
    sc.imsave(os.path.join(blur_path , os.path.basename(img_file)) , img_blur)
 
np.save('images_resize' , images_resize)
np.save('images_blur' , images_blur)

