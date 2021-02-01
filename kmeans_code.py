# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:31:49 2020

@author: bmsri
"""

import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import numpy as np
#import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
#import pickle
import matplotlib.image as mpimg


import os
from sklearn.neighbors import NearestNeighbors
import sklearn.utils as sku
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure, savefig, imshow, axis
from matplotlib.image import imread
import cv2

from sklearn.metrics import accuracy_score

path_to_wd = 'D:\\PR_term_proj\\chest-xray-pneumonia\\chest_xray'
#D:\PR_term_proj\chest-xray-pneumonia\chest_xray
os.chdir(path_to_wd)
WORKING_DIR = os.getcwd()
IMAGE_ORDERING =  "channels_last" 


print('Loading Data ...')

train_im_dir_norm = os.path.join(WORKING_DIR, 'train', 'NORMAL')
validation_im_dir_norm = os.path.join(WORKING_DIR, 'val', 'NORMAL')
test_im_dir_norm = os.path.join(WORKING_DIR, 'test', 'NORMAL')

train_im_dir_pne = os.path.join(WORKING_DIR, 'train', 'PNEUMONIA')
validation_im_dir_pne = os.path.join(WORKING_DIR, 'val','PNEUMONIA')
test_im_dir_pne = os.path.join(WORKING_DIR, 'test','PNEUMONIA')

train_im_list_norm = sorted(os.listdir(train_im_dir_norm))
validation_im_list_norm = sorted(os.listdir(validation_im_dir_norm))
test_im_list_norm = sorted(os.listdir(test_im_dir_norm))

train_im_list_pne = sorted(os.listdir(train_im_dir_pne))
validation_im_list_pne = sorted(os.listdir(validation_im_dir_pne))
test_im_list_pne = sorted(os.listdir(test_im_dir_pne))

TRAIN_IMGS = []
VALIDATION_IMGS = []
TEST_IMGS = []
TRAIN_LABEL = []
VALIDATION_LABEL = []
TEST_LABEL = []



size1 = (200,200)

for i in train_im_list_norm:
    im = Image.open(os.path.join(train_im_dir_norm, i))
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 1)
    if(len(im.shape)==3):
        #print(i)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        #print(im.shape)
    TRAIN_IMGS.append(im)
    TRAIN_LABEL.append(0)


for i in train_im_list_pne:
    im = Image.open(os.path.join(train_im_dir_pne, i))
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 1)
    if(len(im.shape)==3):
        #print(i)
#        imshow(im)
        im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        #print(im.shape)
    TRAIN_IMGS.append(im)
    TRAIN_LABEL.append(1)




for i in validation_im_list_norm:
    im = Image.open(os.path.join(validation_im_dir_norm, i))
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 1)
    VALIDATION_IMGS.append(im)
    VALIDATION_LABEL.append(0)


for i in validation_im_list_pne:
    im = Image.open(os.path.join(validation_im_dir_pne, i))
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 1)
    VALIDATION_IMGS.append(im)
    VALIDATION_LABEL.append(1)




for i in test_im_list_norm:
    im = Image.open(os.path.join(test_im_dir_norm, i))
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 1)
    TEST_IMGS.append(im)
    TEST_LABEL.append(0)


for i in test_im_list_pne:
    im = Image.open(os.path.join(test_im_dir_pne, i))
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 1)
    TEST_IMGS.append(im)
    TEST_LABEL.append(1)



TRAIN_IMGS = np.array(TRAIN_IMGS)
VALIDATION_IMGS = np.array(VALIDATION_IMGS)
TEST_IMGS = np.array(TEST_IMGS)
TRAIN_LABEL = np.array(TRAIN_LABEL)
VALIDATION_LABEL = np.array(VALIDATION_LABEL)
TEST_LABEL = np.array(TEST_LABEL)

TRAIN_IMGS,TRAIN_LABEL = sku.shuffle(TRAIN_IMGS,TRAIN_LABEL,random_state=13)
VALIDATION_IMGS,VALIDATION_LABEL = sku.shuffle(VALIDATION_IMGS,VALIDATION_LABEL,random_state=13)
TEST_IMGS,TEST_LABEL = sku.shuffle(TEST_IMGS,TEST_LABEL,random_state=13)

print(np.shape(TRAIN_IMGS))
print(np.shape(VALIDATION_IMGS))
print(np.shape(TEST_IMGS))



X = TRAIN_IMGS
print('k-means')

output1 = []


img = X
vectorized = img.reshape((np.shape(img)[0],200*200))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
ret_train,label_train,center_train=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

print(np.shape(label_train))
op1 = np.array(output1)
print(op1.shape)
print(X.shape)

score_train = accuracy_score(TRAIN_LABEL, label_train)
print(score_train)


X = VALIDATION_IMGS
print('k-means')

output1 = []

img = X
vectorized = img.reshape((np.shape(img)[0],200*200))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
ret_val,label_val,center_val=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

print(np.shape(label_val))
op1 = np.array(output1)
print(op1.shape)
print(X.shape)

score_val = accuracy_score(VALIDATION_LABEL, label_val)
print(score_val)



X = TEST_IMGS
print('k-means')
output1 = []

img = X
vectorized = img.reshape((np.shape(img)[0],200*200))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
ret_test,label_test,center_test=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
print(np.shape(label_test))
op1 = np.array(output1)
print(op1.shape)
print(X.shape)

score_test = accuracy_score(TEST_LABEL, label_test)
print(score_test)

