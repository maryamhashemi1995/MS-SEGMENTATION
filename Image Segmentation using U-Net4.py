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




TRAIN_PATH = 'C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/train-flaire/images/'
MASK_PATH = 'C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/train-flaire/masks/'
TEST_PATH = 'C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/images/'

train_ids=glob.glob(TRAIN_PATH+"*.png") 
masks_ids=glob.glob(MASK_PATH+"*.png")
test_ids=glob.glob(TEST_PATH+"*.png")


TRAIN_PATH2 = 'C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/train-t2/images/'
MASK_PATH2 = 'C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/train-t2/masks/'
TEST_PATH2 = 'C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-t2/images/'

train_ids2=glob.glob(TRAIN_PATH2+"*.png") 
masks_ids2=glob.glob(MASK_PATH2+"*.png")
test_ids2=glob.glob(TEST_PATH2+"*.png")
#train_ids = next(os.walk(TRAIN_PATH))[1]
#test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1))

X_train2 = np.zeros((len(train_ids2), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train2 = np.zeros((len(train_ids2), IMG_HEIGHT, IMG_WIDTH, 1))

#################################################################################


print('Resizing training images and masks')

n=-1
for image_file in train_ids:
    n=n+1
    img=imread(image_file)
    img=img[40:180,25:160]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_train[n,i,j]=img[i,j]
                
                

mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1)) 
n=-1
for mask_file in masks_ids:
    n=n+1
    mask_ = imread(mask_file)
    mask=mask[40:180,25:160]
    mask_=resize(mask_,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            if mask_[i,j]==0:
                Y_train[n,i,j] =0 
            else:
                Y_train[n,i,j] =255

                
# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
sizes_test = []
print('Resizing test images') 
n=-1
for image_file in test_ids:
    n=n+1
    img=imread(image_file)
    img=img[40:180,25:160]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_test[n,i,j]=img[i,j]

print('Done!')
############################################################

print('Resizing training images and masks')

n=-1
for image_file in train_ids2:
    n=n+1
    img=imread(image_file)
    img=img[40:180,25:160]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_train2[n,i,j]=img[i,j]
                
                

mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1)) 
n=-1
for mask_file in masks_ids2:
    n=n+1
    mask_ = imread(mask_file)
    mask=mask[40:180,25:160]
    mask_=resize(mask_,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            if mask_[i,j]==0:
                Y_train2[n,i,j] =0 
            else:
                Y_train2[n,i,j] =255
                
# test images
X_test2 = np.zeros((len(test_ids2), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
sizes_test = []
print('Resizing test images') 
n=-1
for image_file in test_ids2:
    n=n+1
    img=imread(image_file)
    img=img[40:180,25:160]
    img=resize(img,(IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for i in range (0,IMG_HEIGHT-1):
        for j in range (0,IMG_WIDTH-1):
            X_test2[n,i,j]=img[i,j]

print('Done!')


################################################################
image_x = random.randint(0, len(train_ids))
#image_x2 = random.randint(0, 181)
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()



image_x = random.randint(0, len(train_ids2))
#image_x2 = random.randint(0, 181)
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
##################################################################################
#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
####s=inputs
#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
#c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#c5 = tf.keras.layers.Dropout(0.3)(c5)
#c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
#u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#u6 = tf.keras.layers.concatenate([u6, c4])
#c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#c6 = tf.keras.layers.Dropout(0.2)(c6)
#c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
#u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


model2 = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.summary()
################################################################

#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

#callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        #tf.keras.callbacks.TensorBoard(log_dir='logs')]

history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32 , epochs=20)

history2 = model2.fit(X_train2, Y_train2, validation_split=0.1, batch_size=32 , epochs=20)

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
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


from keras.utils import plot_model
plot_model(model, to_file='model1.png')
####################################################################

training_loss2 = history2.history['loss']
test_loss2 = history2.history['val_loss']

# Create count of the number of epochs
epoch_count2 = range(1, len(training_loss2) + 1)

# Visualize loss history
plt.plot(epoch_count2, training_loss2, 'r--')
plt.plot(epoch_count2, test_loss2, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

from keras.utils import plot_model
plot_model(model2, to_file='model2.png')
#######################################################
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
image=cv2.imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/images/training02_04_flair_pp_new_081.png")
lesion=imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/masks/training02_04_mask1_new_081.png")

image2=cv2.imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-t2/images/training01_02_t2_pp_new_098.png")
lesion2=imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-t2/masks/training01_02_mask2_new_098.png")

imshow(image)
plt.show()
imshow(lesion)
plt.show

imshow(image2)
plt.show()
imshow(lesion2)
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


imaget=cv2.imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/images/training02_04_flair_pp_new_081.png")
lesiont=imread("C:/Users/MaryamHashemi/Desktop/Mahsa/dataset-UNet/test-flaire/masks/training02_04_mask1_new_081.png")

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