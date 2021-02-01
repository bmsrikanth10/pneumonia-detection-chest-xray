import os
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import cv2
import random
import re
import glob
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io as sio

from tensorflow.keras.models import load_model, Model
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import activations

#from vis.utils import utils

from sklearn.manifold import TSNE

saved_model = load_model('baseline_model_without.h5')
#saved_model = load_model('baseline_model_with.h5')
#saved_model = load_model('xnet_model_v1_10.h5')

saved_model.summary()
size1 = (200,200)
#test_files = glob.glob('/export/home/sks6492/PR_Project_Files/new_aug_dir2/test_dir/*/*.png')
test_files = glob.glob('Term_proj/test/*/*.jpeg')
new_test_files = random.sample(test_files, len(test_files))




WORKING_DIR = os.getcwd()
print(os.path.join(WORKING_DIR, 'Term_proj', 'test', 'NORMAL'))
print(test_files)


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
        TEST_LABEL.append(int(0))


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
        TEST_LABEL.append(int(1))




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
        TEST_LABEL.append(int(0))


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
        TEST_LABEL.append(int(1))

TEST_IMGS = np.array(TEST_IMGS)
TEST_LABEL = np.array(TEST_LABEL)

print(np.shape(TEST_IMGS))


X_VAL = TEST_IMGS
Y_VAL = TEST_LABEL

print(np.shape(Y_VAL))
print(np.shape(X_VAL))

saved_model.summary()
size1 = (200,200)
test_files = glob.glob('D:\\PR_term_proj\\chest-xray-pneumonia\\chest_xray\\test\\*\\*.jpeg')
new_test_files = random.sample(test_files, len(test_files))


layer_idx = np.array([6,15,24,33])

from tensorflow.keras import backend as K
def get_activations(model, layer, X_batch):
    get_activation = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activation([X_batch,0])
    return activations

for i in layer_idx:
    activations = get_activations(saved_model, i, X_VAL)
    
    act = np.array(activations)
    
    dimen = act.shape
    
    act = act.reshape(dimen[1],dimen[2]*dimen[3]*dimen[4])
    
    tsne = TSNE(n_components=2, init='pca')
    
    P1_tsne = tsne.fit_transform(act)
    
    classes = ('Normal','Pneumonia')
    plt.figure(1)
    colours = ListedColormap(['g','r'])
    scatter = plt.scatter(P1_tsne[:,0], P1_tsne[:,1], c=Y_VAL, marker='o')#color=['green','red'], 
    #cmap=colours,
    plt.legend(labels=classes)#handles=scatter.legend_elements()[0], labels=classes, fontsize='medium')
    plt.show()
    plt.savefig('Untitled Folder\\t-SNE_train'+str(i)+'.png')

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score 

model = load_model('baseline_model_v1_11.h5')

Y_pred = model.predict(TEST_IMGS, 2)
y_pred_test = np.copy(Y_pred)
y_pred_test[Y_pred>0.5] = 1
y_pred_test[Y_pred<=0.5] = 0

print('Confusion Matrix')
print(confusion_matrix(TEST_LABEL, y_pred_test))
print(accuracy_score(TEST_LABEL, y_pred_test))


