# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:43:36 2020

@author: bmsri
"""

import pandas as pd
import numpy as np
import os
import cv2
import imageio

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

NUM_AUG_IMAGES_WANTED = 10000

IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200

train_image_list = os.listdir('D:\\PR_term_proj\\chest-xray-pneumonia\\chest_xray\\train')

df_train = pd.DataFrame(train_image_list, columns=['image_path'])

df_train.reset_index(inplace=True, drop=True)


    
train_img_path = 'D:\\PR_term_proj\\chest-xray-pneumonia\\chest_xray\\train' 


base_dir = 'base_dir'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)


Normal = os.path.join(train_dir, 'NORMAL')
os.mkdir(Normal)
Pneumonia = os.path.join(train_dir, 'PNEUMONIA')
os.mkdir(Pneumonia)


folder_1 = os.listdir(train_img_path)

train_path_norm = os.path.join(train_img_path, 'NORMAL')
train_path_pne = os.path.join(train_img_path, 'PNEUMONIA')

train_list_norm = sorted(os.listdir(train_path_norm))
train_list_pne = sorted(os.listdir(train_path_pne))


comp_path = 'D:\\PR_term_proj\\'

#df_data.set_index('image_path', inplace=True)
from PIL import Image

for image in train_list_norm:  
    fname = image
    #label = df_data.loc[image, 'target']
    
    #if fname in folder_1:
        
    src = train_path_norm + '\\' + fname
    dst = comp_path + train_dir + '\\' + "NORMAL" + '\\' + fname
    
    image = Image.open(src)
    image=image.resize([200,200])
    print(np.shape(image))
    if (len(np.shape(image)) == 2):
        image.save(dst)
    

for image in train_list_pne:  
    fname = image
        
    src = train_path_pne + '\\' + fname
    dst = comp_path + train_dir + '\\' + "PNEUMONIA" + '\\' + fname
        
    image = Image.open(src)
#    print(np.shape(image))
    image=image.resize([200,200])
    print(np.shape(image))
    if (len(np.shape(image)) == 2):
        image.save(dst)


'''
Data Augmentation
'''     
aug_dir = 'aug_dir'
os.mkdir(aug_dir)
   
class_list = ['NORMAL','PNEUMONIA']

for item in class_list:
    
         
    img_class = item

    img_list = sorted(os.listdir(comp_path + 'base_dir\\train_dir\\' + img_class))

    os.mkdir(aug_dir+ "\\" + item)
##
    for fname in img_list:
            src = os.path.join(comp_path + 'base_dir\\train_dir\\' + img_class, fname)
            dst = os.path.join(aug_dir+ "\\" + item, fname)
            shutil.copyfile(src, dst)

    path = aug_dir+ "\\"# + item + "\\"
    save_path = comp_path + 'base_dir\\train_dir\\' + img_class

    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#rescale=1/255, 
    batch_size = 50
    
    aug_datagen = datagen.flow_from_directory(path, save_to_dir=aug_dir+ "\\" + item + "\\", save_format='jpg', target_size=(IMAGE_HEIGHT,IMAGE_WIDTH), batch_size=batch_size)
    
    num_files = len(os.listdir(aug_dir+ "\\" + item))
    
    num_batches = int(np.ceil((NUM_AUG_IMAGES_WANTED-num_files)/batch_size))

    for i in range(0,num_batches):
        #imgs = Image.open(aug_datagen)
        imgs, labels = next(aug_datagen)
        print(np.shape(imgs))

    
