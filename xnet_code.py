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

#path_to_wd = ''
#D:\PR_term_proj\chest-xray-pneumonia\chest_xray
#os.chdir(path_to_wd)
WORKING_DIR = os.getcwd()
IMAGE_ORDERING =  "channels_last" 


print('Loading Data ...')

train_im_dir_norm = os.path.join(WORKING_DIR, 'Term_proj','aug_dir','NORMAL')
train_im_dir_pne = os.path.join(WORKING_DIR, 'Term_proj','aug_dir','PNEUMONIA')

train_im_list_norm = sorted(os.listdir(train_im_dir_norm))
train_im_list_pne = sorted(os.listdir(train_im_dir_pne))

TRAIN_IMGS = []
TRAIN_LABEL = []

size1 = (200,200)

for i in train_im_list_norm:
    im = cv2.imread(os.path.join(train_im_dir_norm, i))
    #im = Image.open(os.path.join(train_im_dir_norm, i))
    #im = Image.open(os.path.join(train_im_dir_norm, i))
    #im = im.resize([200,200])

    im = np.array(im)
    #a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(im, 255)
    if(len(im.shape)==3):
        #print(i)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        #print(im.shape)
        TRAIN_IMGS.append(im)
        TRAIN_LABEL.append(0)


for i in train_im_list_pne:
    im = cv2.imread(os.path.join(train_im_dir_pne, i))
    #im = Image.open(os.path.join(train_im_dir_pne, i))
    #im = Image.open(os.path.join(train_im_dir_pne, i))
    #im = im.resize([200,200])
    im = np.array(im)
    #a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(im, 255)
    if(len(im.shape)==3):
        #print(i)
        #imshow(im)
        #im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        #print(im.shape)
        TRAIN_IMGS.append(im)
        TRAIN_LABEL.append(1)

TRAIN_IMGS = np.array(TRAIN_IMGS)
#VALIDATION_IMGS = np.array(VALIDATION_IMGS)
#TEST_IMGS = np.array(TEST_IMGS)
TRAIN_LABEL = np.array(TRAIN_LABEL)

del train_im_list_norm
del train_im_list_pne
del im

print(np.shape(TRAIN_IMGS))


validation_im_dir_norm = os.path.join(WORKING_DIR, 'Term_proj', 'val', 'NORMAL')
test_im_dir_norm = os.path.join(WORKING_DIR, 'Term_proj', 'test', 'NORMAL')

validation_im_dir_pne = os.path.join(WORKING_DIR, 'Term_proj', 'val','PNEUMONIA')
test_im_dir_pne = os.path.join(WORKING_DIR, 'Term_proj', 'test','PNEUMONIA')

validation_im_list_norm = sorted(os.listdir(validation_im_dir_norm))
test_im_list_norm = sorted(os.listdir(test_im_dir_norm))

validation_im_list_pne = sorted(os.listdir(validation_im_dir_pne))
test_im_list_pne = sorted(os.listdir(test_im_dir_pne))

VALIDATION_IMGS = []
TEST_IMGS = []
VALIDATION_LABEL = []
TEST_LABEL = []


for i in validation_im_list_norm:
    #im = Image.open(os.path.join(validation_im_dir_norm, i))
    im = cv2.imread(os.path.join(validation_im_dir_norm, i))
#    im = Image.open(os.path.join(validation_im_dir_norm, i))
#    im = im.resize([200,200])
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 255)
    if(len(im.shape)==3):
        #print(i)
#        imshow(im)
     #   im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        TEST_IMGS.append(im)
        TEST_LABEL.append(0)


for i in validation_im_list_pne:
    #im = Image.open(os.path.join(validation_im_dir_pne, i))
    im = cv2.imread(os.path.join(validation_im_dir_pne, i))
#    im = Image.open(os.path.join(validation_im_dir_pne, i))
#    im = im.resize([200,200])
    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 255)
    if(len(im.shape)==3):
        #print(i)
#        imshow(im)
     #   im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        TEST_IMGS.append(im)
        TEST_LABEL.append(1)




for i in test_im_list_norm:
    #im = Image.open(os.path.join(test_im_dir_norm, i))
    im = cv2.imread(os.path.join(test_im_dir_norm, i))
#    im = Image.open(os.path.join(test_im_dir_norm, i))
#    im = im.resize([200,200])

    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 255)
    if(len(im.shape)==3):
        #print(i)
#        imshow(im)
     #   im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        TEST_IMGS.append(im)
        TEST_LABEL.append(0)


for i in test_im_list_pne:
    #im = Image.open(os.path.join(test_im_dir_pne, i))
    im = cv2.imread(os.path.join(test_im_dir_pne, i))
#    im = Image.open(os.path.join(test_im_dir_pne, i))
    #im = im.resize([200,200])

    im = np.array(im)
    a = cv2.resize(im, size1, interpolation = cv2.INTER_AREA)
    im = np.divide(a, 255)
    if(len(im.shape)==3):
        #print(i)
#        imshow(im)
     #   im = (0.2125*im[:,:,0] + 0.7154*im[:,:,1] + 0.0721*im[:,:,2])
        TEST_IMGS.append(im)
        TEST_LABEL.append(1)

del validation_im_list_norm
del validation_im_list_pne
del test_im_list_norm
del test_im_list_pne
del im
#del a

TEST_IMGS = np.array(TEST_IMGS)
TEST_LABEL = np.array(TEST_LABEL)


print(np.shape(TRAIN_IMGS))
print(np.shape(TEST_IMGS))


imshow(TRAIN_IMGS[0])


imshow(TEST_IMGS[0])

TRAIN_IMGS = TRAIN_IMGS.reshape(np.shape(TRAIN_IMGS)[0], 200, 200, 3)
#VALIDATION_IMGS = VALIDATION_IMGS.reshape(np.shape(VALIDATION_IMGS)[0], 200, 200, 1)
TEST_IMGS = TEST_IMGS.reshape(np.shape(TEST_IMGS)[0], 200, 200, 3)


img_input = Input(shape = (200,200,3), name = 'image_input')

## Block 1
x1_1 = Conv2D(4, (3, 3), padding='same', name='block1_conv1_pre', dilation_rate = 9, data_format=IMAGE_ORDERING )(img_input)
x1_2 = Conv2D(4, (5, 5), padding='same', name='block1_conv2_pre', dilation_rate = 9, data_format=IMAGE_ORDERING )(img_input)
x1_3 = Conv2D(4, (7, 7), padding='same', name='block1_conv3_pre', dilation_rate = 9, data_format=IMAGE_ORDERING )(img_input)
x1_4 = Conv2D(4, (9, 9), padding='same', name='block1_conv4_pre', dilation_rate = 9, data_format=IMAGE_ORDERING )(img_input)
#x = Conv2D(128, (3, 3), padding='same', name='block4_conv2_post', dilation_rate = 2, data_format=IMAGE_ORDERING )(x)
x = Concatenate(name='concatenate1')([x1_1, x1_2, x1_3, x1_4])
x = Activation('relu')(x)
x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_post', data_format=IMAGE_ORDERING )(x)
x = Dropout(0.3)(x)
f1_out = x

# CONCATENATE

#com = Concatenate(name='concatenate')([pool4_post, pool4_pre])


#x = Activation('relu')(x)x = Conv2D(32, (3, 3), padding='valid', name='block1_conv1_pre', data_format=IMAGE_ORDERING )(img_input)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_pre', data_format=IMAGE_ORDERING )(x)
#x = Dropout(0.10)(x)
#f1_pre = x

## Block 2
x2_1 = Conv2D(8, (3, 3), padding='same', name='block2_conv1_pre', dilation_rate = 7, data_format=IMAGE_ORDERING )(x)
x2_2 = Conv2D(8, (5, 5), padding='same', name='block2_conv2_pre', dilation_rate = 7, data_format=IMAGE_ORDERING )(x)
x2_3 = Conv2D(8, (7, 7), padding='same', name='block2_conv3_pre', dilation_rate = 7, data_format=IMAGE_ORDERING )(x)
x2_4 = Conv2D(8, (9, 9), padding='same', name='block2_conv4_pre', dilation_rate = 7, data_format=IMAGE_ORDERING )(x)
#x = Conv2D(128, (3, 3), padding='same', name='block4_conv2_post', dilation_rate = 2, data_format=IMAGE_ORDERING )(x)
x = Concatenate(name='concatenate2')([x2_1, x2_2, x2_3, x2_4])
x = Activation('relu')(x)
x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_post', data_format=IMAGE_ORDERING )(x)
x = Dropout(0.3)(x)
f2_out = x


## Block 3
x3_1 = Conv2D(16, (3, 3), padding='same', name='block3_conv1_pre', dilation_rate = 5, data_format=IMAGE_ORDERING )(x)
x3_2 = Conv2D(16, (5, 5), padding='same', name='block3_conv2_pre', dilation_rate = 5, data_format=IMAGE_ORDERING )(x)
x3_3 = Conv2D(16, (7, 7), padding='same', name='block3_conv3_pre', dilation_rate = 5, data_format=IMAGE_ORDERING )(x)
x3_4 = Conv2D(16, (9, 9), padding='same', name='block3_conv4_pre', dilation_rate = 5, data_format=IMAGE_ORDERING )(x)
#x = Conv2D(128, (3, 3), padding='same', name='block4_conv2_post', dilation_rate = 2, data_format=IMAGE_ORDERING )(x)
x = Concatenate(name='concatenate3')([x3_1, x3_2, x3_3, x3_4])
x = Activation('relu')(x)
x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_post', data_format=IMAGE_ORDERING )(x)
x = Dropout(0.3)(x)
f3_out = x


## Block 4
x4_1 = Conv2D(32, (3, 3), padding='same', name='block4_conv1_pre', dilation_rate = 3, data_format=IMAGE_ORDERING )(x)
x4_2 = Conv2D(32, (5, 5), padding='same', name='block4_conv2_pre', dilation_rate = 3, data_format=IMAGE_ORDERING )(x)
x4_3 = Conv2D(32, (7, 7), padding='same', name='block4_conv3_pre', dilation_rate = 3, data_format=IMAGE_ORDERING )(x)
x4_4 = Conv2D(32, (9, 9), padding='same', name='block4_conv4_pre', dilation_rate = 3, data_format=IMAGE_ORDERING )(x)
#x = Conv2D(128, (3, 3), padding='same', name='block4_conv2_post', dilation_rate = 2, data_format=IMAGE_ORDERING )(x)
x = Concatenate(name='concatenate4')([x4_1, x4_2, x4_3, x4_4])
x = Activation('relu')(x)
x = BatchNormalization()(x)
#x = Activation('relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_post', data_format=IMAGE_ORDERING )(x)
x = Dropout(0.3)(x)
f4_out = x


# Block 2
#x = Conv2D(64, (3, 3), padding='valid', name='block2_conv1_pre', data_format=IMAGE_ORDERING )(x)
#x = Activation('relu')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_pre', data_format=IMAGE_ORDERING )(x)
#x = Dropout(0.2)(x)
#f2_pre = x

# Block 3
#x = Conv2D(128, (3, 3), padding='valid', name='block3_conv1_pre', data_format=IMAGE_ORDERING )(x)
#x = Activation('relu')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_pre', data_format=IMAGE_ORDERING )(x)
#x = Dropout(0.2)(x)
#pool3_pre = x

# Block 4
#x = Conv2D(128, (3, 3), padding='valid', name='block4_conv1_pre', data_format=IMAGE_ORDERING )(x)
#x = Activation('relu')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_pre', data_format=IMAGE_ORDERING )(x)
#pool4_pre = x

x = Flatten()(x)
x = Dropout(0.25)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(100, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(25, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=img_input, outputs=output)
model.summary()


#model.load_weights("xnet_model_v1_10.h5")

opt = optimizers.Adam(lr=1E-5)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

hist1 = model.fit(TRAIN_IMGS,TRAIN_LABEL,
                  validation_data=(TEST_IMGS,TEST_LABEL),
                  batch_size=20,epochs=120,verbose=1)

#score = model.evaluate(TEST_IMGS,TEST_LABEL,  batch_size=2)

#print(score)
model.save('xnet_model_v1_11.h5')
model.save_weights('xnet_model_v1_11_weights.h5')
model_json = model.to_json()

with open('xnet_model_v1_11.json', 'w') as json_file:
                  json_file.write(model_json)


plt.plot(hist1.history['acc'])
plt.plot(hist1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score 

Y_pred = model.predict(TEST_IMGS, 2)
y_pred_test = np.copy(Y_pred)
y_pred_test[Y_pred>0.5] = 1
y_pred_test[Y_pred<=0.5] = 0

print('Confusion Matrix')
print(confusion_matrix(TEST_LABEL, y_pred_test))
print(accuracy_score(TEST_LABEL, y_pred_test))

Y_pred = model.predict(TRAIN_IMGS, 2)
y_pred_train = np.copy(Y_pred)
y_pred_train[Y_pred>0.5] = 1
y_pred_train[Y_pred<=0.5] = 0

print('Confusion Matrix')
print(confusion_matrix(TRAIN_LABEL, y_pred_train))

print(accuracy_score(TRAIN_LABEL, y_pred_train))






















