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
from keras_lr_finder.lr_finder import LRFinder
import keras.backend as k

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
#IMG_WIDTH = 256
#IMG_HEIGHT = 256
IMG_CHANNELS = 3




#################################################################################
TRAIN_PATH1 = 'E:/.../train-flaire/images/'
MASK_PATH1 = 'E:/.../train-flaire/masks/'
TEST_PATH1 = 'E:/.../test-flaire/images/'
Mask_PATH2 = 'E:/.../test-flaire/masks/'

train_ids=glob.glob(TRAIN_PATH1+"*.png") 
masks_ids=glob.glob(MASK_PATH1+"*.png")
test_ids=glob.glob(TEST_PATH1+"*.png")
masks_ids2=glob.glob(Mask_PATH2+"*.png")


################################################################################# 
############  PREPROCESSING PART ############  

           
Y_train = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, 1))
#1238 is number of masks that are not total black and have some information to be learned
n=-1
index=0
d=0
index_=[]
zerocount=0
notzero=0
for mask_file in masks_ids:
    index=index+1
    mask= imread(mask_file)
    mask=mask[30:187,11:168]
    mask=resize(mask,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if (np.all(mask==0)):
        d=d+1
    else:
        n=n+1
        index_.append(index)
        for i in range (0,IMG_HEIGHT):
            for j in range (0,IMG_WIDTH):
                if mask[i,j]==0:
                    Y_train[n,i,j] =0 
                    zerocount=zerocount+1
                else:
                    Y_train[n,i,j] =1
                    notzero=notzero+1

#generally we have 128*128*1237 pixels. some them are white(lesion) and others are black(background)
# calculation the proporsion of black to white pixles
print("zerocount",zerocount,"notzero",notzero)    
print("all pixels",(zerocount+notzero),"black/white pixels",(zerocount/notzero))


                    
                    
X_train = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#print('Resizing training images and masks')
n=-1
for i in index_:
    n=n+1
    img=imread(train_ids[i])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT):
        for j in range (0,IMG_WIDTH):
            X_train[n,i,j]=img[i,j]


#################################################################################                
# test images

Y_test = np.zeros((86, IMG_HEIGHT, IMG_WIDTH, 1))

n=-1
index=0
d=0
index_=[]
zerocount_t=0
notzero_t=0
for mask_file in masks_ids2:
    index=index+1
    mask= imread(mask_file)
    mask=mask[30:187,11:168]
    mask=resize(mask,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if (np.all(mask==0)):
        d=d+1
    else:
        n=n+1
        index_.append(index)
        for i in range (0,IMG_HEIGHT):
            for j in range (0,IMG_WIDTH):
                if mask[i,j]==0:
                    Y_test[n,i,j] =0
                    zerocount_t=zerocount_t+1
                else:
                    Y_test[n,i,j] =1
                    notzero_t=notzero_t+1

                   
#generally we have 128*128*1237 pixels. some them are white(lesion) and others are black(background)
# calculation the proporsion of black to white pixles
print("zerocount",zerocount_t,"notzero",notzero_t)    
print("all pixels",(zerocount_t+notzero_t),"black/white pixels",(zerocount_t/notzero_t))                    
                    
                    
                    
X_test = np.zeros((86, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

n=-1
for i in index_:
    n=n+1
    img=imread(test_ids[i])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT):
        for j in range (0,IMG_WIDTH):
            X_test[n,i,j]=img[i,j]


print('Done!')

    

################################################################
#Just show an example image to make sure we have our images
#image_x = random.randint(0, len(X_train))
#imshow(X_train[image_x])
#plt.show()
#imshow(np.squeeze(Y_train[image_x]))
#plt.show()


##################################################################################

#Build the model


import tensorflow as tf
from tensorflow.keras import backend as K                                       
from tensorflow.keras.layers import Dropout, SpatialDropout2D, Conv2D,\
                                    Conv2DTranspose, MaxPooling2D, concatenate,\
                                    ELU, BatchNormalization, Activation, \
                                    ZeroPadding2D, multiply, Lambda, UpSampling2D,\
                                    Add, Multiply
from tensorflow.keras import Model, Input





def DSC_Evaluator(y_true, y_pred):
    
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
    return DSC


def generalized_dice_loss(y_true, y_pred):
    
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



import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

    
    
def custom_loss(y_true, y_pred):
    
    
    return 1.5*generalized_dice_loss(y_true, y_pred) +  binary_crossentropy(y_true, y_pred)
    






image_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)


def Attention_U_Net_2D(image_shape, activation,
                       feature_maps, depth,
                       drop_values, spatial_dropout, 
                       batch_norm, k_init, loss_type, 
                       optimizer, lr, n_classes):
   

    if len(feature_maps) != depth+1:
        raise ValueError("feature_maps dimension must be equal depth+1")
    if len(drop_values) != depth+1:
        raise ValueError("'drop_values' dimension must be equal depth+1")

    dinamic_dim = (None,)*(len(image_shape)-1) + (image_shape[-1],)
    x = Input(dinamic_dim)                                                     
    #x = Input(image_shape)                                                     
    inputs = x
        
    if loss_type == "w_bce":
        weights = Input(image_shape)

    # List used to access layers easily to make the skip connections of the U-Net
    l=[]

    # ENCODER
    for i in range(depth):
        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)
        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x 
        x = Activation(activation) (x)

        l.append(x)
    
        x = MaxPooling2D((2, 2))(x)

    # BOTTLENECK
    x = Conv2D(feature_maps[depth], (3, 3), activation=None,
               kernel_initializer=k_init, padding='same')(x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)
    if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[depth]) (x)
            else:
                x = Dropout(drop_values[depth]) (x)
    x = Conv2D(feature_maps[depth], (3, 3), activation=None,
               kernel_initializer=k_init, padding='same') (x)
    x = BatchNormalization() (x) if batch_norm else x
    x = Activation(activation) (x)

    # DECODER
    for i in range(depth-1, -1, -1):
        x = Conv2DTranspose(feature_maps[i], (2, 2), 
                            strides=(2, 2), padding='same') (x)
        attn = AttentionBlock(x, l[i], feature_maps[i], batch_norm)
        x = concatenate([x, attn])
        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)
        if drop_values is not None:
            if spatial_dropout:
                x = SpatialDropout2D(drop_values[i]) (x)
            else:
                x = Dropout(drop_values[i]) (x)

        x = Conv2D(feature_maps[i], (3, 3), activation=None,
                   kernel_initializer=k_init, padding='same') (x)
        x = BatchNormalization() (x) if batch_norm else x
        x = Activation(activation) (x)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid') (x)
    
    # Loss type
    if loss_type == "w_bce":
        model = Model(inputs=[inputs, weights], outputs=[outputs])
    else:
        model = Model(inputs=[inputs], outputs=[outputs])

    # Select the optimizer
    if optimizer == "sgd":
        opt =  "sgd"
        #tf.keras.optimizers.SGD(lr=lr, momentum=0.99, decay=0.0, nesterov=False)
    elif optimizer == "adam":
        opt="adam"
        #opt=tf.keras.optimizers.RMSprop(lr=0.01)
        #opt= tf.keras.optimizers.Adagrad( lr=0.0001)
        #opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,amsgrad=False)
    else:
        raise ValueError("Error: optimizer value must be 'sgd' or 'adam'")

    # Compile the model
    if loss_type == "bce":
        if n_classes > 1:
            model.compile(optimizer=opt, loss='categorical_crossentropy',
                          metrics=[DSC_Evaluator])
        else:
            #model.compile(optimizer=opt, loss='binary_crossentropy',metrics=[Mean_IOU_Evaluator])
            model.compile(optimizer=opt, loss=custom_loss ,metrics=[DSC_Evaluator])
    else:
        raise ValueError("'loss_type' must be 'bce', 'w_bce' or 'w_bce_dice'")

    return model




def AttentionBlock(x, shortcut, filters, batch_norm):


    g1 = Conv2D(filters, kernel_size = 1)(shortcut) 
    g1 = BatchNormalization() (g1) if batch_norm else g1
    x1 = Conv2D(filters, kernel_size = 1)(x) 
    x1 = BatchNormalization() (x1) if batch_norm else x1

    g1_x1 = Add()([g1,x1])
    psi = Activation('elu')(g1_x1)
    psi = Conv2D(1, kernel_size = 1)(psi) 
    psi = BatchNormalization() (psi) if batch_norm else psi
    psi = Activation('sigmoid')(psi)
    x = Multiply()([x,psi])
    return x

##################################################################################

                                    
model=Attention_U_Net_2D(image_shape,'elu',
                       [16, 32, 64, 128, 256],4,
                       [0.1,0.1,0.2,0.2,0.3], False,False,'he_normal',"bce", 
                       "adam", 0.005, 1)

##################################################################################


results = model.fit(X_train, Y_train, validation_split=0.15, batch_size=16, epochs=300)
####################################

