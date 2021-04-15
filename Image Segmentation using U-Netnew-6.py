# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:24:14 2020

@author: MaryamHashemi
"""

import tensorflow as tf

import os
import random
import numpy as np
import cv2 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


#################################################################################
#from google.colab import drive
#drive.mount('/content/drive')
#!unzip "/content/drive/My Drive/Colab Notebooks/dataset-UNet.zip"

#################################################################################
TRAIN_PATH1 = 'E:/CODE/Mahsa/dataset-UNet/train-flaire/images/'
MASK_PATH1 = 'E:/CODE/Mahsa/dataset-UNet/train-flaire/masks/'
TEST_PATH1 = 'E:/CODE/Mahsa/dataset-UNet/test-flaire/images/'
Mask_PATH2 = 'E:/CODE/Mahsa/dataset-UNet/test-flaire/masks/'

train_ids=glob.glob(TRAIN_PATH1+"*.png") 
masks_ids=glob.glob(MASK_PATH1+"*.png")
test_ids=glob.glob(TEST_PATH1+"*.png")
masks_ids2=glob.glob(Mask_PATH2+"*.png")



#################################################################################
TRAIN_PATH1 = '/content/drive/My Drive/data-MS/train-flaire/images/'
MASK_PATH1 = '/content/drive/My Drive/data-MS/train-flaire/masks/'
TEST_PATH1 = '/content/drive/My Drive/data-MS/test-flaire/images/'
Mask_PATH2 = '/content/drive/My Drive/data-MS/test-flaire/masks/'

print(TRAIN_PATH1)
train_ids=glob.glob(TRAIN_PATH1+"*.png") 
masks_ids=glob.glob(MASK_PATH1+"*.png")
test_ids=glob.glob(TEST_PATH1+"*.png")
masks_ids2=glob.glob(Mask_PATH2+"*.png")

print(train_ids)
           
#################################################################################            
Y_train = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, 1))
#1238 is number of masks that are not total black and have some information to be learned
n=-1
index=0
d=0
index_=[]
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
        for i in range (0,IMG_HEIGHT-1):
            for j in range (0,IMG_WIDTH-1):
                if mask[i,j]==0:
                    Y_train[n,i,j] =0 
                else:
                    Y_train[n,i,j] =1


X_train = np.zeros((1237, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#print('Resizing training images and masks')
n=-1
for i in index_:
    n=n+1
    img=imread(train_ids[i])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_train[n,i,j]=img[i,j]/255


#################################################################################                
# test images


Y_test = np.zeros((87, IMG_HEIGHT, IMG_WIDTH, 1))

n=0
index=0
d=0
index_=[]
for mask_file in masks_ids2:
    index=index+1
    mask= imread(mask_file)
    mask=mask[30:187,11:168]
    mask_=resize(mask,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if (np.all(mask==0)):
        d=d+1
    else:
        n=n+1
        index_.append(index)
        for i in range (0,IMG_HEIGHT-1):
            for j in range (0,IMG_WIDTH-1):
                if mask[i,j]==0:
                    Y_test[n,i,j] =0 
                else:
                    Y_test[n,i,j] =1
                    
                    
                    
                    
                    
X_test = np.zeros((87, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

n=-1
for i in index_:
    n=n+1
    img=imread(test_ids[i])
    img=img[30:187,11:168]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_test[n,i,j]=img[i,j]/255


print('Done!')

    

################################################################


#image_x = random.randint(0, len(X_train))
#image_x2 = random.randint(0, 181)
#imshow(X_train[image_x])
#plt.show()
#imshow(np.squeeze(Y_train[image_x]))
#plt.show()


##################################################################################
#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
s=inputs
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



import keras
import keras.backend as k
from tensorflow.compat.v1.keras import backend as K


def Mean_IOU_Evaluator(y_true, y_pred):   
    prec = []
    
    for t in np.arange(0.5, 1, 0.05):
        
        #y_pred_ = tf.to_int32(y_pred>t)
        y_pred_=tf.cast((y_pred>t), tf.int32)
        #score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        score, up_opt = tf.compat.v1.metrics.mean_iou (y_true, y_pred_, 2)
        #k.get_session().run(tf.local_variables_initializer())
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return k.mean(k.stack(prec), axis = 0)



def DSC(y_true, y_pred):   
    prec = []
    
    for t in np.arange(0.5, 1, 0.05):
        
        #y_pred_ = tf.to_int32(y_pred>t)
        y_pred_=tf.cast((y_pred>t), tf.int32)
        #score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        score, up_opt = tf.compat.v1.metrics.mean_iou (y_true, y_pred_, 2)
        #k.get_session().run(tf.local_variables_initializer())
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
            score_dsc=(2*score)/(score+1)
        prec.append(score_dsc)
    return k.mean(k.stack(prec), axis = 0)


#NEW= tf.compat.v1.metrics.mean_iou(num_classes=2)    
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Mean_IOU_Evaluator])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[DSC])
#model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[DSC])
model.summary()






# 7. Show The Results per Epoch


  
################################################################

#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

#callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        #tf.keras.callbacks.TensorBoard(log_dir='logs')]

history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=2 , epochs=150)

#history2 = model2.fit(X_train2, Y_train2, validation_split=0.1, batch_size=32 , epochs=20)

##############################################################################################
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Intersection Over Union')
plt.ylabel('iou')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
####################################################################
plt.plot(history.history['DSC'])
plt.plot(history.history['val_DSC'])
plt.title('dsc')
plt.ylabel('DSC')
plt.xlabel('epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()

from keras.utils import plot_model
plot_model(model, to_file='model1.png')
####################################################################


idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.95).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

#####################################################################
#image=cv2.imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/trainnew/images/training04_04_flair_pp_new_082.png")
image=cv2.imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/images/training02_04_flair_pp_new_064.png")
lesion=imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/masks/training02_04_mask1_new_064.png")


imshow(image)
plt.show()
imshow(lesion)
plt.show



image=resize(image,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
imagee = np.expand_dims(image, axis=0)
predict=model.predict(imagee)

predictnew=np.zeros( (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for i in range (0,predictnew.shape[0]-1):
    for j in range (0,predictnew.shape[1]-1):
        if predict[0,i,j,0]==1:
        #print(predict[0,i,j,0])
            predictnew[i,j]=predict[0,i,j,0]

        
imshow(predict[0,:,:,0])        
imshow(predictnew)

cv2.imshow('',predictnew)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("result.jpg", predictnew)




image2=resize(image2,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
imagee2 = np.expand_dims(image2, axis=0)
predict2=model2.predict(imagee2)

predictnew2=np.zeros( (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for i in range (0,predictnew2.shape[0]-1):
    for j in range (0,predictnew2.shape[1]-1):
        if predict2[0,i,j,0]==1:
        #print(predict[0,i,j,0])
            predictnew2[i,j]=predict2[0,i,j,0]

        
        
imshow(predictnew2)

##############################################


imaget=cv2.imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/images/training02_04_flair_pp_new_132.png")
lesiont=imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/masks/training02_04_mask1_new_132.png")

imshow(imaget)
plt.show()
imshow(lesiont)
plt.show


imaget=resize(imaget,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
imageet = np.expand_dims(imaget, axis=0)
predictf=model.predict(imageet)
predictt2=model2.predict(imageet)

predictnewt=np.zeros( (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for i in range (0,predictnewt.shape[0]-1):
    for j in range (0,predictnewt.shape[1]-1):
        if predictf[0,i,j,0]==1 or predictt2[0,i,j,0]==1:
        #print(predict[0,i,j,0])
            predictnewt[i,j]=predictf[0,i,j,0]
            
            
imshow(predictnewt)  
imshow(predictt2[0,:,:,0])         
imshow(predictf[0,:,:,0])   