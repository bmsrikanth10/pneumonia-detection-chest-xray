# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:29:15 2020

@author: bmsri
"""

"""
Created on Mon Feb 24 23:57:05 2020

@author: bmsri
"""

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
    TEST_IMGS.append(im)
    TEST_LABEL.append(0)


for i in validation_im_list_pne:
    im = Image.open(os.path.join(validation_im_dir_pne, i))
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 1)
    TEST_IMGS.append(im)
    TEST_LABEL.append(1)




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
#VALIDATION_IMGS = np.array(VALIDATION_IMGS)
TEST_IMGS = np.array(TEST_IMGS)
TRAIN_LABEL = np.array(TRAIN_LABEL)
#VALIDATION_LABEL = np.array(VALIDATION_LABEL)
TEST_LABEL = np.array(TEST_LABEL)

TRAIN_IMGS,TRAIN_LABEL = sku.shuffle(TRAIN_IMGS,TRAIN_LABEL,random_state=13)
#VALIDATION_IMGS,VALIDATION_LABEL = sku.shuffle(VALIDATION_IMGS,VALIDATION_LABEL,random_state=13)
TEST_IMGS,TEST_LABEL = sku.shuffle(TEST_IMGS,TEST_LABEL,random_state=13)

print(np.shape(TRAIN_IMGS))
#print(np.shape(VALIDATION_IMGS))
print(np.shape(TEST_IMGS))

TRAIN_IMGS = TRAIN_IMGS.reshape(np.shape(TRAIN_IMGS)[0], 200*200)
#VALIDATION_IMGS = VALIDATION_IMGS.reshape(np.shape(VALIDATION_IMGS)[0], 200*200)
TEST_IMGS = TEST_IMGS.reshape(np.shape(TEST_IMGS)[0], 200*200)

X = TRAIN_IMGS

print('nearest neighbors')
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X)
distances_test, indices_test = nbrs.kneighbors(TEST_IMGS)
#distances_val, indices_val = nbrs.kneighbors(VALIDATION_IMGS)




for i in range(np.shape(TEST_IMGS)[0]):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1)
    im1 = Image.fromarray(TEST_IMGS[i].reshape(200,200))
    ax1.imshow(im1)
    ax1.axis('off')
    ax1.set_title('Test Image')
    
    ax2 = fig.add_subplot(1,2,2)
    im2 = Image.fromarray(TRAIN_IMGS[indices_test[i]].reshape(200,200))
    ax2.imshow(im2)
    ax2.axis('off')
    ax2.set_title('Closest Train Image')
    
    fig.suptitle('Test Image and Closest Train image, distance' + str(distances_test[i]), fontsize=16)
    fig.savefig(os.path.join(WORKING_DIR,'Test_Train_data_study')+'\\knn_test_train_'+str(i)+'.jpeg')
    plt.close('all')

#print(np.mean(distances_val))
print(np.mean(distances_test))



neighbors_train_list = []
for i in range(len(indices_test)):
    neighbors_train_list.append(TRAIN_IMGS[indices_test[i]])
neighbors_train = np.array(neighbors_train_list)
neighbors_train = neighbors_train.reshape(640,40000)
print(np.shape(neighbors_train))
print(np.shape(TEST_IMGS))


abs_diff = np.abs(neighbors_train - TEST_IMGS)
l1 = np.sum(abs_diff,1)
l1 = l1.reshape(640,1)
print(abs_diff.shape)
print(l1.shape)
print(distances_test.shape)


print('mean of distance')
print(np.mean(l1))
print('median of distance')
print(np.median(l1))
print('standard deviation of distance')
print(np.std(l1))
print('median of distance')
print(np.median(l1))
print('minimum of distance')
print(np.min(l1))
print('maximum of distance')
print(np.max(l1))


print('mean of distance')
print(np.mean(distances_test))
print('median of distance')
print(np.median(distances_test))
print('standard deviation of distance')
print(np.std(distances_test))
print('median of distance')
print(np.median(distances_test))
print('minimum of distance')
print(np.min(distances_test))
print('maximum of distance')
print(np.max(distances_test))