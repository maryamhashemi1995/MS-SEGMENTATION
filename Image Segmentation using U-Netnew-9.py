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
            X_train[n,i,j]=img[i,j]


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
            X_test[n,i,j]=img[i,j]


print('Done!')

    

################################################################


#image_x = random.randint(0, len(X_train))
#image_x2 = random.randint(0, 181)
#imshow(X_train[image_x])
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
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
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
model.summary()



#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_IOU])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Mean_IOU_Evaluator])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[DSC])
#model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[DSC])
#model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[Mean_IOU_Evaluator])
#model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])



checkpointer = tf.keras.callbacks.ModelCheckpoint('model_h5_checkpoint', verbose = 1, save_best_only=True)
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]




results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=15, callbacks=callbacks)


# 7. Show The Results per Epoch

####################################

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()
 

ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
 


ix = random.randint(0, len(preds_test_t))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(Y_test[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()



##############################################################################################
training_loss = results.history['loss']
test_loss = results.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

#plt.figure()
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Intersection Over Union')
#plt.ylabel('iou')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
####################################################################
plt.plot(history.history['mean_io_u'])
plt.plot(history.history['val_mean_io_u'])
plt.title('dsc')
plt.ylabel('DSC')
plt.xlabel('epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()

from keras.utils import plot_model
plot_model(model, to_file='model1.png')
####################################################################



for i in range(preds_test_t.shape[0]):
    for m in range(preds_test_t.shape[1]):
        for n in range(preds_test_t.shape[2]):
            if preds_test[i,m,n,0]>0.3:
                #print(preds_test[i,m,n,0])
                preds_test[i,m,n,0]=1
            else:
                #print(preds_test[i,m,n,0])
                preds_test[i,m,n,0]=0
                


TP_array=[]
TN_array=[]
FP_array=[]
FN_array=[]
for i in range(preds_test.shape[0]):
    TP=0
    TN=0
    FP=0
    FN=0    
    for m in range(preds_test.shape[1]):
        for n in range(preds_test.shape[2]):
            if preds_test[i,m,n,0]==Y_test[i,m,n,0]==1:
                TP=TP+1
            if preds_test[i,m,n,0]==Y_test[i,m,n,0]==0:
                TN=TN+1
            if preds_test[i,m,n,0]==0 and Y_test[i,m,n,0]==1:
                FN=FN+1
            if preds_test[i,m,n,0]==1 and Y_test[i,m,n,0]==0:
                FP=FP+1
    TP_array.append(TP) 
    FP_array.append(FP)
    TN_array.append(TN)
    FN_array.append(FN)    


def parameters(TP_array,TN_array,FP_array,FN_array):
    
    DSC_array=[]
    Jac_array=[]
    prec_array=[]
    accu_array=[]
    sens_array=[]
    speci_array=[]
    for i in range(preds_test.shape[0]):
        jac=TP_array[i]/(TP_array[i]+FP_array[i]+FN_array[i])
        Jac_array.append(jac)
    return Jac_array
     
####################################################################
def DSC(y_true, y_pred):   
    prec = []
    
    for t in np.arange(0.5, 1, 0.05):
        
        #y_pred_ = tf.to_int32(y_pred>t)
        #y_pred_=tf.cast((y_pred>t), tf.int32)
        y_pred_=tf.cast(y_pred, tf.int32)
        y_true_=tf.cast(Y_test, tf.int32)
        #score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        score, up_opt = tf.compat.v1.metrics.mean_iou (y_true, y_pred_, 2)
        #k.get_session().run(tf.local_variables_initializer())
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
            score_dsc=(2*score)/(score+1)
        prec.append(score_dsc)
    return k.mean(k.stack(prec), axis = 0)


DSC(Y_test,preds_test)

tf.keras.metrics.Accuracy(
    name='accuracy', dtype=None
)
tf.keras.metrics.FalseNegatives(
    thresholds=None, name=None, dtype=None
)
tf.keras.metrics.FalsePositives(
    thresholds=None, name=None, dtype=None
)

tf.keras.metrics.TrueNegatives(
    thresholds=None, name=None, dtype=None
)

tf.keras.metrics.TruePositives(
    thresholds=None, name=None, dtype=None
)

tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)

tfma.metrics.Specificity(
    thresholds: Optional[List[float]] = None,
    name: Text = SPECIFICITY_NAME
)

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

