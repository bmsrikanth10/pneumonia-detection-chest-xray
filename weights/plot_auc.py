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

layer_idx = np.array([6,15,24,33])#utils.find_layer_idx(saved_model,'conv2d_3')
lr_probs = saved_model.predict(X_VAL)

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(Y_VAL, lr_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot

Y_pred = saved_model.predict(X_VAL, 2)
y_pred_test = np.copy(Y_pred)
y_pred_test[Y_pred>0.5] = 1
y_pred_test[Y_pred<=0.5] = 0

lr_precision, lr_recall, _ = precision_recall_curve(Y_VAL,lr_probs )
lr_f1, lr_auc = f1_score(Y_VAL, y_pred_test), auc(lr_recall, lr_precision)
# summarize scores
print('Results: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(Y_VAL[Y_VAL==1]) / len(Y_VAL)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='our model')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
