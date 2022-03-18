# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:24:14 2020

@author: MaryamHashemi
"""

#import libraries
import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)

import os
import random
import numpy as np
import cv2 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob

import keras.backend as k

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
#IMG_WIDTH = 256
#IMG_HEIGHT = 256
IMG_CHANNELS = 3


#################################################################################
#from google.colab import drive
#drive.mount('/content/drive')
#!unzip "/content/drive/My Drive/Colab Notebooks/dataset-UNet.zip"

#################################################################################
#reading MRI scans
############# Comment by mahsa
# TRAIN_PATH_t2 = 'E:/CODE/Mahsa/dataset-UNet/train-t2/images/'
# TRAINMASK_PATH_t2 = 'E:/CODE/Mahsa/dataset-UNet/train-t2/masks/'
# TEST_PATH_t2 = 'E:/CODE/Mahsa/dataset-UNet/test-t2/images/'
# TESTMask_PATH_t2 = 'E:/CODE/Mahsa/dataset-UNet/test-t2/masks/'

# TRAIN_PATH_flair = 'E:/CODE/Mahsa/dataset-UNet/train-flaire/images/'
# TRAINMASK_PATH_flair = 'E:/CODE/Mahsa/dataset-UNet/train-flaire/masks/'
# TEST_PATH_flair = 'E:/CODE/Mahsa/dataset-UNet/test-flaire/images/'
# TESTMask_PATH_flair= 'E:/CODE/Mahsa/dataset-UNet/test-flaire/masks/'


#reading MRI scans   (add by Mahsa)
TRAIN_PATH_t2 = 'E:/MS_Python_18Far/T2-FLAIR/train-t2/images/'
TRAINMASK_PATH_t2 = 'E:/MS_Python_18Far/T2-FLAIR/train-t2/masks/'
TEST_PATH_t2 = 'E:/MS_Python_18Far/T2-FLAIR/test-t2/images/'
TESTMask_PATH_t2 = 'E:/MS_Python_18Far/T2-FLAIR/test-t2/masks/'

TRAIN_PATH_flair = 'E:/MS_Python_18Far/T2-FLAIR/train-flaire/images/'
TRAINMASK_PATH_flair = 'E:/MS_Python_18Far/T2-FLAIR/train-flaire/masks/'
TEST_PATH_flair = 'E:/MS_Python_18Far/T2-FLAIR/test-flaire/images/'
TESTMask_PATH_flair= 'E:/MS_Python_18Far/T2-FLAIR/test-flaire/masks/'




train_ids_t2=glob.glob(TRAIN_PATH_t2+"*.png") 
trainmasks_ids_t2=glob.glob(TRAINMASK_PATH_t2+"*.png")
test_ids_t2=glob.glob(TEST_PATH_t2+"*.png")
testmasks_ids_t2=glob.glob(TESTMask_PATH_t2+"*.png")

train_ids_flair=glob.glob(TRAIN_PATH_flair+"*.png") 
trainmasks_ids_flair=glob.glob(TRAINMASK_PATH_flair+"*.png")
test_ids_flair=glob.glob(TEST_PATH_flair+"*.png")
testmasks_ids_flair=glob.glob(TESTMask_PATH_flair+"*.png")

           
################################################################################# 
############  PREPROCESSING PART ############              
Y_train1 = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, 1))
#1237 is number of masks that are not total black and have some information to be learned
n=-1
index=0
d=0
index_=[]


for mask_file in trainmasks_ids_t2:
    index=index+1
    mask= imread(mask_file)
    mask=mask[30:187,11:168]
    mask=resize(mask,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if (np.all(mask==0)):
        d=d+1
    else:
        n=n+1
        index_.append(index)
        for i in range (0,IMG_HEIGHT-1):
            for j in range (0,IMG_WIDTH-1):
                if mask[i,j]==0:
                    Y_train1[n,i,j] =0 
                else:
                    Y_train1[n,i,j] =1
                    
print(n)

print(d)                    

X_train1 = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#print('Resizing training images and masks')
n=-1
for i in index_:
    n=n+1
    img=imread(train_ids_t2[i])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_train1[n,i,j]=img[i,j]
            
            
###################################

Y_train2 = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, 1))
#1237 is number of masks that are not total black and have some information to be learned
n=-1
index=0
d=0
index_=[]

for mask_file in trainmasks_ids_flair:
    index=index+1
    mask= imread(mask_file)
    mask=mask[30:187,11:168]
    mask=resize(mask,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if (np.all(mask==0)):
        d=d+1
    else:
        n=n+1
        index_.append(index)
        for i in range (0,IMG_HEIGHT-1):
            for j in range (0,IMG_WIDTH-1):
                if mask[i,j]==0:
                    Y_train2[n,i,j] =0 
                else:
                    Y_train2[n,i,j] =1                    



X_train2 = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#print('Resizing training images and masks')
n=-1
for i in index_:
    n=n+1
    img=imread(train_ids_flair[i])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_train2[n,i,j]=img[i,j]
 
##########################################
Y_train = np.zeros((1237+1237, IMG_HEIGHT, IMG_WIDTH, 1))

for n in range(0,Y_train.shape[0]-1):
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            if n<1237:
                Y_train[n,i,j]=Y_train1[n,i,j]
            else:
                Y_train[n,i,j]=Y_train2[n-1237,i,j]




X_train = np.zeros((1237+1237, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

for n in range(0,X_train.shape[0]-1):
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            if n<1237:
                X_train[n,i,j]=X_train1[n,i,j]
            else:
                X_train[n,i,j]=X_train2[n-1237,i,j]               
 
    
 
#################################################################################                
# test images
############## 
Y_test1 = np.zeros((86, IMG_HEIGHT, IMG_WIDTH, 1))

n=-1
d=0
index=0
index_=[]
for mask_file in testmasks_ids_t2:
    index=index+1
    mask= imread(mask_file)
    mask=mask[30:187,11:168]
    mask=resize(mask,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if (np.all(mask==0)):
        d=d+1
    else:
        n=n+1
        index_.append(index)
        for i in range (0,IMG_HEIGHT-1):
            for j in range (0,IMG_WIDTH-1):
                if mask[i,j]==0:
                    Y_test1[n,i,j] =0 
                else:
                    Y_test1[n,i,j] =1
                    
print(n)
print(d)                    

 

X_test1 = np.zeros((86, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

n=-1

for p in index_: 
    n=n+1
    img=imread(test_ids_t2[p])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_test1[n,i,j]=img[i,j]

                   
###############################
                    
Y_test2 = np.zeros((86, IMG_HEIGHT, IMG_WIDTH, 1))

n=-1
d=0
index=0
index_=[]
for mask_file in testmasks_ids_flair:
    index=index+1
    mask= imread(mask_file)
    mask=mask[30:187,11:168]
    mask=resize(mask,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if (np.all(mask==0)):
        d=d+1
    else:
        index_.append(index)
        n=n+1
        for i in range (0,IMG_HEIGHT-1):
            for j in range (0,IMG_WIDTH-1):
                if mask[i,j]==0:
                    Y_test2[n,i,j] =0 
                else:
                    Y_test2[n,i,j] =1                    
print(n)                    
print(d)  
                    
  

 
X_test2 = np.zeros((86, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

n=-1

for p in index_: 
    n=n+1
    img=imread(test_ids_flair[p])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_test2[n,i,j]=img[i,j]



print('Done!')

    

################################################################
  #####################                 
Y_test = np.zeros((86+86, IMG_HEIGHT, IMG_WIDTH, 1))

for n in range(0,Y_test.shape[0]-1):
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            if n<86:
                Y_test[n,i,j]=Y_test1[n,i,j]
            else:
                Y_test[n,i,j]=Y_test2[n-86,i,j]




X_test = np.zeros((86+86, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

for n in range(0,X_test.shape[0]-1):
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            if n<86:
                X_test[n,i,j]=X_test1[n,i,j]
            else:
                X_test[n,i,j]=X_test2[n-86,i,j]    
                
                
                
#########################################################
#Just show an example image to make sure we have our images
#image_x = random.randint(0, len(X_train))
#imshow((X_train[image_x])/256)
#plt.show()
#imshow(np.squeeze(Y_train[image_x]))
#plt.show()


##################################################################################
#from sklearn.datasets import make_classification
#from sklearn.model_selection import train_test_split
#Y_test,Y_train,X_test, X_train  = train_test_split(Y_train, X_train, test_size=0.93, shuffle=True)



##################################################################################
#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
#s=inputs
#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.summary()


###################################################################################### 

#define metric and loss functions

def DSC_Evaluator(y_true, y_pred):
    
    prec = []
    
    for t in np.arange(0.5, 1, 0.05):
        
        y_pred_ = tf.to_int32(y_pred>t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        k.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
        IOU= k.mean(k.stack(prec), axis = 0)
        DSC=(2*IOU)/(IOU+1)
        return DSC

import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

def generalized_DSC_loss(y_true, y_pred):
    
    prec = []
    
    for t in np.arange(0.5, 1, 0.05):
        
        y_pred_ = tf.to_int32(y_pred>t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        k.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
        IOU=k.mean(k.stack(prec), axis = 0)
        DSC=(2*IOU)/(IOU+1)
        Loss=1-DSC
    return Loss

    
#  ############### Comment by Mahsa   
# def custom_loss(y_true, y_pred):
    
    
#     return 1.5*generalized_DSC_loss(y_true, y_pred) +  binary_crossentropy(y_true, y_pred)
    

################### Add by Mahsa
def custom_loss(y_true, y_pred):
  

    bb=binary_crossentropy(y_true, y_pred)
    gg= 1.5*generalized_DSC_loss(y_true, y_pred)
    return tf.cast(gg,dtype='float64') +  tf.cast(bb,dtype='float64')

######################################################################################
#Train the model


opt=tf.keras.optimizers.Adam(lr=0.005)
model.compile(optimizer="adam", loss=custom_loss, 
                  metrics=[DSC_Evaluator])

#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_h5_checkpoint', verbose = 1, save_best_only=True)
#callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
       # tf.keras.callbacks.TensorBoard(log_dir='logs')]




#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=30, callbacks=callbacks)
results = model.fit(X_train, Y_train, validation_split=0.15, batch_size=16, epochs=300)

####################################
