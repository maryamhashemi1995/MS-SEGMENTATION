# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:24:14 2020

@author: MaryamHashemi
"""

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

########## Add by Mahsa
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
TRAIN_PATH1 = 'E:/CODE/Mahsa/dataset-UNet/train-flaire/images/'
MASK_PATH1 = 'E:/CODE/Mahsa/dataset-UNet/train-flaire/masks/'
TEST_PATH1 = 'E:/CODE/Mahsa/dataset-UNet/test-flaire/images/'
Mask_PATH2 = 'E:/CODE/Mahsa/dataset-UNet/test-flaire/masks/'

#TRAIN_PATH1 = 'E:/MS_Python_18Far/train-flaire/images/'
#MASK_PATH1 = 'E:/MS_Python_18Far/train-flaire/masks/'
#TEST_PATH1 = 'E:/MS_Python_18Far/test-flaire/images/'
#Mask_PATH2 = 'E:/MS_Python_18Far/test-flaire/masks/'



train_ids=glob.glob(TRAIN_PATH1+"*.png") 
masks_ids=glob.glob(MASK_PATH1+"*.png")
test_ids=glob.glob(TEST_PATH1+"*.png")
masks_ids2=glob.glob(Mask_PATH2+"*.png")



#################################################################################
#TRAIN_PATH1 = '/content/drive/My Drive/data-MS/train-flaire/images/'
#MASK_PATH1 = '/content/drive/My Drive/data-MS/train-flaire/masks/'
#TEST_PATH1 = '/content/drive/My Drive/data-MS/test-flaire/images/'
#Mask_PATH2 = '/content/drive/My Drive/data-MS/test-flaire/masks/'

#print(TRAIN_PATH1)
#train_ids=glob.glob(TRAIN_PATH1+"*.png") 
#masks_ids=glob.glob(MASK_PATH1+"*.png")
#test_ids=glob.glob(TEST_PATH1+"*.png")
#masks_ids2=glob.glob(Mask_PATH2+"*.png")

#print(train_ids)
           
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

n=-1
index=0
d=0
index_=[]
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


#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_IOU])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Mean_IOU_Evaluator])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[DSC])
#model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[DSC])
#model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[Mean_IOU_Evaluator])
#model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
####### comment by Mahsa
#Adam=tf.keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,
#    name="Adam")

############ Comment by Mahsa
#model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])



#### Add by Mahsa
def Mean_IOU_Evaluator(y_true, y_pred):
    
    prec = []
    
    for t in np.arange(0.5, 1, 0.05):
        
        y_pred_ = tf.to_int32(y_pred>t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        k.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return k.mean(k.stack(prec), axis = 0)


############ Written by Mahsa
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Mean_IOU_Evaluator])

opt=tf.keras.optimizers.Adam(lr=0.005)
model.compile(optimizer="adam", loss='binary_crossentropy', 
                  metrics=[Mean_IOU_Evaluator])

#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_h5_checkpoint', verbose = 1, save_best_only=True)
#callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
       # tf.keras.callbacks.TensorBoard(log_dir='logs')]




#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=30, callbacks=callbacks)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=200)
# 7. Show The Results per Epoch

####################################
############## Add by Mahsa

# 11.1. Summarize History for Loss

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()


# 11.1. Summarize History for IOU

plt.plot(results.history['Mean_IOU_Evaluator'])
plt.plot(results.history['val_Mean_IOU_Evaluator'])
plt.title('Intersection Over Union')
plt.ylabel('IOU')
plt.xlabel('epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()



##############################################################################################

from keras.utils import plot_model
plot_model(model, to_file='model_flaire_challenge2015.png')
# Save the weights
model.save_weights('model_weights_flaire_challenge2015.h5')

# Save the model architecture
with open('model_architecture_flaire_challenge2015.json', 'w') as f:
    f.write(model.to_json())
    
model.save('model_flaire_challenge2015.h5')

################################################################################# 

 
from matplotlib import pyplot as plt
idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
plt.subplot(1,3,1)
plt.imshow(X_train[ix]/256)
plt.title('MRI Scan (Train Set)')

plt.subplot(1,3,2)
plt.imshow(np.squeeze(Y_train[ix]))
plt.title('Original Mask (Train Set)')

plt.subplot(1,3,3)
plt.imshow(np.squeeze(preds_train_t[ix]))
plt.title('Predicted Mask (Train Set)')

 ###########################  

ix = random.randint(0, len(preds_val_t))
plt.subplot(1,3,1)
plt.imshow((X_train[int(X_train.shape[0]*0.9):][ix])/256)
plt.title('MRI Scan (Validation Set)')
plt.subplot(1,3,2)
plt.imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.title('Original Mask (Validation Set)')
plt.subplot(1,3,3)
plt.imshow(np.squeeze(preds_val_t[ix]))
plt.title('Predicted Mask (Validation Set)')
 


 ########################### 
ix = random.randint(0, len(preds_test_t))
plt.subplot(1,3,1)
plt.imshow((X_test[ix])/256)
plt.title('MRI Scan (Test Set)')

plt.subplot(1,3,2)
plt.imshow(np.squeeze(Y_test[ix]))
plt.title('Original Mask (Test Set)')

plt.subplot(1,3,3)
plt.imshow(np.squeeze(preds_test_t[ix]))
plt.title('Predicted Mask (Test Set)')


####################################################################


def parameters(TP_array,TN_array,FP_array,FN_array):
    
    DSC_array=[]
    accu_array=[]
    sens_array=[]
    speci_array=[]
    errorrate_array=[]
    fnr_array=[]
    exfra_array=[]
    ppv_array=[]
    npv_array=[]
    for i in range(0,preds_test.shape[0]-1):
        DSC=(2*TP_array[i])/((2*TP_array[i])+FN_array[i]+FP_array[i])
        Accuracy= (TP_array[i]+TN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i]) 
        Sensitivity_me=(TP_array[i])/(TP_array[i]+FN_array[i])
        Specificity_me=(TN_array[i])/(TN_array[i]+FP_array[i])
        Error_rate=(FP_array[i]+FN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i])
        FNR=(FN_array[i])/(TN_array[i]+FN_array[i])
        Extra_Fraction=(FP_array[i])/(TN_array[i]+FN_array[i])
        NPV=(TN_array[i])/(TN_array[i]+FN_array[i])
        if TP_array[i]==FP_array[i]==0:
            PPV=0.33
        else:
            PPV=(TP_array[i])/(TP_array[i]+FP_array[i])
        
        DSC_array.append(DSC)
        accu_array.append(Accuracy)
        sens_array.append(Sensitivity_me)
        speci_array.append(Specificity_me)
        errorrate_array.append(Error_rate)
        fnr_array.append(FNR)
        exfra_array.append(Extra_Fraction)
        ppv_array.append(PPV)
        npv_array.append(NPV)
    return DSC_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array




TP_array=[]
TN_array=[]
FP_array=[]
FN_array=[]
for i in range(preds_test_t.shape[0]):
    TP=0
    TN=0
    FP=0
    FN=0    
    for m in range(preds_test_t.shape[1]):
        for n in range(preds_test_t.shape[2]):
            if preds_test_t[i,m,n,0]==Y_test[i,m,n,0]==1:
                TP=TP+1
            if preds_test_t[i,m,n,0]==Y_test[i,m,n,0]==0:
                TN=TN+1
            if preds_test_t[i,m,n,0]==0 and Y_test[i,m,n,0]==1:
                FN=FN+1
            if preds_test_t[i,m,n,0]==1 and Y_test[i,m,n,0]==0:
                FP=FP+1
    TP_array.append(TP) 
    FP_array.append(FP)
    TN_array.append(TN)
    FN_array.append(FN)    


DSC_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array =parameters(TP_array,TN_array,FP_array,FN_array)



cul1=0
cul2=0
cul3=0
cul4=0
cul5=0
cul6=0
cul7=0
cul8=0
cul9=0
for i in range(0,preds_test.shape[0]-1):
    cul1=DSC_array[i]+cul1
    cul2=accu_array[i]+cul2
    cul3=sens_array[i]+cul3
    cul4=speci_array[i]+cul4
    cul5=errorrate_array[i]+cul5
    cul6=fnr_array[i]+cul6
    cul7=exfra_array[i]+cul7
    cul8=ppv_array[i]+cul8
    cul9=npv_array[i]+cul9
    
    
average_DSC=cul1/(preds_test.shape[0])
average_accuracy=cul2/(preds_test.shape[0])
average_sensitivity=cul3/(preds_test.shape[0])
average_specificity=cul4/(preds_test.shape[0])
average_errorrate=cul5/(preds_test.shape[0])
average_fnr=cul6/(preds_test.shape[0])
average_exfra=cul7/(preds_test.shape[0])
average_ppv=cul8/(preds_test.shape[0])
average_npv=cul9/(preds_test.shape[0])

print("average_DSC",average_DSC,"average_accuracy",average_accuracy,"average_sensitivity",average_sensitivity,"average_specificity",average_specificity)
print("average_Errore_Rate",average_errorrate,"average_FNR",average_fnr,"average_Extra_Fraction",average_exfra,"average_PPV",average_ppv,"average_npv",average_npv)


####################################################################

def parameters2(TP_array,TN_array,FP_array,FN_array):
    
    DSC_array=[]
    accu_array=[]
    sens_array=[]
    speci_array=[]
    errorrate_array=[]
    fnr_array=[]
    exfra_array=[]
    ppv_array=[]
    npv_array=[]
    for i in range(0,preds_train.shape[0]-1):
        DSC=(2*TP_array[i])/((2*TP_array[i])+FN_array[i]+FP_array[i])
        Accuracy= (TP_array[i]+TN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i]) 
        Sensitivity_me=(TP_array[i])/(TP_array[i]+FN_array[i])
        Specificity_me=(TN_array[i])/(TN_array[i]+FP_array[i])
        Error_rate=(FP_array[i]+FN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i])
        FNR=(FN_array[i])/(TN_array[i]+FN_array[i])
        Extra_Fraction=(FP_array[i])/(TN_array[i]+FN_array[i])
        NPV=(TN_array[i])/(TN_array[i]+FN_array[i])
        if TP_array[i]==FP_array[i]==0:
            PPV=0
        else:
            PPV=(TP_array[i])/(TP_array[i]+FP_array[i])
        
        
        DSC_array.append(DSC)
        accu_array.append(Accuracy)
        sens_array.append(Sensitivity_me)
        speci_array.append(Specificity_me)
        errorrate_array.append(Error_rate)
        fnr_array.append(FNR)
        exfra_array.append(Extra_Fraction)
        ppv_array.append(PPV)
        npv_array.append(NPV)
    return DSC_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array


TP1_array=[]
TN1_array=[]
FP1_array=[]
FN1_array=[]
for i in range(preds_train.shape[0]):
    TP=0
    TN=0
    FP=0
    FN=0    
    for m in range(preds_train.shape[1]):
        for n in range(preds_train.shape[2]):
            if preds_train_t[i,m,n,0]==Y_train[i,m,n,0]==1:
                TP=TP+1
            if preds_train_t[i,m,n,0]==Y_train[i,m,n,0]==0:
                TN=TN+1
            if preds_train_t[i,m,n,0]==0 and Y_train[i,m,n,0]==1:
                FN=FN+1
            if preds_train_t[i,m,n,0]==1 and Y_train[i,m,n,0]==0:
                FP=FP+1
    TP1_array.append(TP) 
    FP1_array.append(FP)
    TN1_array.append(TN)
    FN1_array.append(FN)    
    
    
    
DSC_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array =parameters2(TP1_array,TN1_array,FP1_array,FN1_array)



cul1=0
cul2=0
cul3=0
cul4=0
cul5=0
cul6=0
cul7=0
cul8=0
cul9=0
for i in range(0,preds_train.shape[0]-1):
    cul1=DSC_array[i]+cul1
    cul2=accu_array[i]+cul2
    cul3=sens_array[i]+cul3
    cul4=speci_array[i]+cul4
    cul5=errorrate_array[i]+cul5
    cul6=fnr_array[i]+cul6
    cul7=exfra_array[i]+cul7
    cul8=ppv_array[i]+cul8
    cul9=npv_array[i]+cul9
    
    
average_DSC=cul1/(preds_train.shape[0]-1)
average_accuracy=cul2/(preds_train.shape[0]-1)
average_sensitivity=cul3/(preds_train.shape[0]-1)
average_specificity=cul4/(preds_train.shape[0]-1)
average_errorrate=cul5/(preds_train.shape[0]-1)
average_fnr=cul6/(preds_train.shape[0]-1)
average_exfra=cul7/(preds_train.shape[0]-1)
average_ppv=cul8/(preds_train.shape[0]-1)
average_npv=cul9/(preds_train.shape[0]-1)
print("average_DSC",average_DSC,"average_accuracy",average_accuracy,"average_sensitivity",average_sensitivity,"average_specificity",average_specificity)
print("average_Errore_Rate",average_errorrate,"average_FNR",average_fnr,"average_Extra_Fraction",average_exfra,"average_PPV",average_ppv,"average_npv",average_npv)



####################################################################
from matplotlib import colors
ix = random.randint(0, len(preds_test_t)-1)

a=Y_test[ix]
b=preds_test_t[ix]
eshterak = np.zeros((IMG_HEIGHT, IMG_WIDTH))
ejtema= np.zeros((IMG_HEIGHT, IMG_WIDTH))
notditectedlesion=np.zeros((IMG_HEIGHT, IMG_WIDTH))
wronglesiondetected=np.zeros((IMG_HEIGHT, IMG_WIDTH))

for i in range(a.shape[0]-1):
    for j in range(a.shape[1]-1):
        if a[i,j]==b[i,j]:
            eshterak[i,j]=a[i,j]
            ejtema[i,j]=a[i,j]
        else:
            if a[i,j]==1 or b[i,j]==1:
                ejtema[i,j]=1
                if a[i,j]==1 and b[i,j]==0:
                    notditectedlesion[i,j]=1
                if a[i,j]==0 and b[i,j]==1:
                    wronglesiondetected[i,j]=1
                
            


plt.subplot(2,4,1)
plt.imshow((X_test[ix])/256)
plt.title('MRI Scan (Test Set)')

plt.subplot(2,4,2)
plt.imshow(np.squeeze(Y_test[ix]))
plt.title('Original Mask (Test Set)')

plt.subplot(2,4,3)
plt.imshow(np.squeeze(preds_test_t[ix]))
plt.title('Predicted Mask (Test Set)')

plt.subplot(2,4,4)
plt.imshow(np.squeeze(ejtema))
plt.title('Predicted+Original Mask')

plt.subplot(2,4,5)
plt.imshow(np.squeeze(eshterak))
plt.title('|Predicted-Original| Mask')

plt.subplot(2,4,6)
plt.imshow(np.squeeze(notditectedlesion))
#Lesion Area that Network did not Detect
plt.title('False Negative')

plt.subplot(2,4,7)
plt.imshow(np.squeeze(wronglesiondetected))
#Not Lesion Area that Network Classified as Lesion 
plt.title('False Positive')
                    
plt.subplot(2,4,8)
plt.imshow(notditectedlesion,cmap=colors.ListedColormap(['black', 'blue']))
plt.imshow(wronglesiondetected,cmap=colors.ListedColormap(['black', 'red']),alpha=0.6)
plt.title("Blue=False Negative, Red=False Positive")
#################################################################### 


#Accuracy=(TP+TN)/(TP+TN+FN+FP) 
#DSC=(2*TP)/(2*TP+FN+FP)
#Error_rate=(FP+FN)/(TP+TN+FN+FP) 
#Sensitivity=(TP)/(TP+FN)
#Specificity=(TN)/(TN+FP)
#FPR=1-Specificity
#FNR=(FN)/(TN+FN)
#Extra_Fraction=(FP)/(TN+FN)
#PPV=(TP)/(TP+FP)