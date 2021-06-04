# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:49:48 2021

@author: MaryamHashemi

"""

import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)
import numpy as np
import cv2 
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.utils import np_utils
import glob
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from skimage.transform import resize
from keras.models import model_from_json
import random
seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

with open("C:/Users/MaryamHashemi/Desktop/MS/Results/attentionUNet-DSC-costumloss-adam/Flaire-metric=DSC,opt=adam,loss=costumloss-1.5-1,epochs=300/model_architecture_flaire__attentionUNET_challenge2015_Adam_Cstumloss-1.5-1_DSC.json", 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
     
#     Load weights into the new model
model.load_weights('C:/Users/MaryamHashemi/Desktop/MS/Results/attentionUNet-DSC-costumloss-adam/Flaire-metric=DSC,opt=adam,loss=costumloss-1.5-1,epochs=300/model_weights_flaire_attentionUNET_challenge2015_Adam_Cstumloss-1.5-1_DSC.h5')



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


####################################################################

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.85)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.85):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

####################################################################
################################################################################# 
#calculate metrics and performance for test data


def parameters(TP_array,TN_array,FP_array,FN_array):
    counter_array=[]
    DSC_array=[]
    IOU_array=[]
    accu_array=[]
    sens_array=[]
    speci_array=[]
    errorrate_array=[]
    fnr_array=[]
    exfra_array=[]
    ppv_array=[]
    npv_array=[]
    count=0
    for i in range(0,preds_test.shape[0]-1):
        DSC=(2*TP_array[i])/((2*TP_array[i])+FN_array[i]+FP_array[i])
        IOU=(TP_array[i])/((TP_array[i])+FN_array[i]+FP_array[i])
        Accuracy= (TP_array[i]+TN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i]) 
        Sensitivity_me=(TP_array[i])/(TP_array[i]+FN_array[i])
        Specificity_me=(TN_array[i])/(TN_array[i]+FP_array[i])
        Error_rate=(FP_array[i]+FN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i])
        FNR=(FN_array[i])/(TN_array[i]+FN_array[i])
        Extra_Fraction=(FP_array[i])/(TN_array[i]+FN_array[i])
        NPV=(TN_array[i])/(TN_array[i]+FN_array[i])
        PPV=(TP_array[i])/(TP_array[i]+FP_array[i])
        
        count=count+1
        counter_array.append(count)
        IOU_array.append(IOU)
        DSC_array.append(DSC)
        accu_array.append(Accuracy)
        sens_array.append(Sensitivity_me)
        speci_array.append(Specificity_me)
        errorrate_array.append(Error_rate)
        fnr_array.append(FNR)
        exfra_array.append(Extra_Fraction)
        ppv_array.append(PPV)
        npv_array.append(NPV)
    return counter_array,DSC_array,IOU_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array




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


counter_array,DSC_array,IOU_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array =parameters(TP_array,TN_array,FP_array,FN_array)


import statistics
print("Mean of DSC for Test data is:", (statistics.mean(DSC_array)))
print("Variance of DSC for Test data is:", (statistics.variance(DSC_array)))

print("Mean of IOU for Test data is:", (statistics.mean(IOU_array)))
print("Variance of IOU for Test data is:", (statistics.variance(IOU_array)))

print("Mean of Accuracy for Test data is:", (statistics.mean(accu_array)))
print("Variance of Accuracy for Test data is:", (statistics.variance(accu_array)))

print("Mean of Sensitivity for Test data is:", (statistics.mean(sens_array)))
print("Variance of Sensitivity for Test data is:", (statistics.variance(sens_array)))

print("Mean of Specificity for Test data is:", (statistics.mean(speci_array)))
print("Variance of Specificity for Test data is:", (statistics.variance(speci_array)))

print("Mean of Error Rate for Test data is:", (statistics.mean(errorrate_array)))
print("Variance of Error Rate for Test data is:", (statistics.variance(errorrate_array)))

print("Mean of FNR for Test data is:", (statistics.mean(fnr_array)))
print("Variance of FNR for Test data is:", (statistics.variance(fnr_array)))

print("Mean of Extra Fraction for Test data is:", (statistics.mean(exfra_array)))
print("Variance of Extra Fraction for Test data is:", (statistics.variance(exfra_array)))

print("Mean of PPV for Test data is:", (statistics.mean(ppv_array)))
print("Variance of PPV for Test data is:", (statistics.variance(ppv_array)))

print("Mean of NPV for Test data is:", (statistics.mean(npv_array)))
print("Variance of NPV for Test data is:", (statistics.variance(npv_array)))


plt.title("DSC Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,DSC_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("DSC",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('DSC_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("IOU Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,IOU_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("IOU",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('IOU_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Accuracy Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,accu_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Accuracy",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Accuracy_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Sensitivity Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,sens_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Sensitivity",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Sensitivity_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Specificity Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,speci_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Specificity",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Specificity_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Error Rate Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,errorrate_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Error Rate",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Error Rate_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("FNR Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,fnr_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("FNR",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('FNR_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Extra Fraction Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,exfra_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Extra Fraction",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Extra Fraction_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("PPV Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,ppv_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("PPV",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('PPV_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("NPV Scatter Plot for Test Data", fontsize='16')	#title
plt.scatter( counter_array,npv_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("NPV",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('NPV_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()
####################################################################
#calculate metrics and performance for train data

def parameters2(TP_array,TN_array,FP_array,FN_array):
    counter_array=[]
    DSC_array=[]
    IOU_array=[]
    accu_array=[]
    sens_array=[]
    speci_array=[]
    errorrate_array=[]
    fnr_array=[]
    exfra_array=[]
    ppv_array=[]
    npv_array=[]
    count=0
    for i in range(0,preds_train.shape[0]-1):
        DSC=(2*TP_array[i])/((2*TP_array[i])+FN_array[i]+FP_array[i])
        IOU=(TP_array[i])/((TP_array[i])+FN_array[i]+FP_array[i])
        Accuracy= (TP_array[i]+TN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i]) 
        Sensitivity_me=(TP_array[i])/(TP_array[i]+FN_array[i])
        Specificity_me=(TN_array[i])/(TN_array[i]+FP_array[i])
        Error_rate=(FP_array[i]+FN_array[i])/(TP_array[i]+TN_array[i]+FN_array[i]+FP_array[i])
        FNR=(FN_array[i])/(TN_array[i]+FN_array[i])
        Extra_Fraction=(FP_array[i])/(TN_array[i]+FN_array[i])
        NPV=(TN_array[i])/(TN_array[i]+FN_array[i])
        PPV=(TP_array[i])/(TP_array[i]+FP_array[i])
        
        count=count+1
        counter_array.append(count)
        IOU_array.append(IOU)
        DSC_array.append(DSC)
        accu_array.append(Accuracy)
        sens_array.append(Sensitivity_me)
        speci_array.append(Specificity_me)
        errorrate_array.append(Error_rate)
        fnr_array.append(FNR)
        exfra_array.append(Extra_Fraction)
        ppv_array.append(PPV)
        npv_array.append(NPV)
    return counter_array,DSC_array,IOU_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array



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
    
    
    
counter_array,DSC_array,IOU_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array =parameters2(TP1_array,TN1_array,FP1_array,FN1_array)



import statistics
print("Mean of DSC for Train data is:", (statistics.mean(DSC_array)))
print("Variance of DSC for Train data is:", (statistics.variance(DSC_array)))

print("Mean of IOU for Train data is:", (statistics.mean(IOU_array)))
print("Variance of IOU for Train data is:", (statistics.variance(IOU_array)))

print("Mean of Accuracy for Train data is:", (statistics.mean(accu_array)))
print("Variance of Accuracy for Train data is:", (statistics.variance(accu_array)))

print("Mean of Sensitivity for Train data is:", (statistics.mean(sens_array)))
print("Variance of Sensitivity for Train data is:", (statistics.variance(sens_array)))

print("Mean of Specificity for Train data is:", (statistics.mean(speci_array)))
print("Variance of Specificity for Train data is:", (statistics.variance(speci_array)))

print("Mean of Error Rate for Train data is:", (statistics.mean(errorrate_array)))
print("Variance of Error Rate for Train data is:", (statistics.variance(errorrate_array)))

print("Mean of FNR for Train data is:", (statistics.mean(fnr_array)))
print("Variance of FNR for Train data is:", (statistics.variance(fnr_array)))

print("Mean of Extra Fraction for Train data is:", (statistics.mean(exfra_array)))
print("Variance of Extra Fraction for Train data is:", (statistics.variance(exfra_array)))

print("Mean of PPV for Train data is:", (statistics.mean(ppv_array)))
print("Variance of PPV for Train data is:", (statistics.variance(ppv_array)))

print("Mean of NPV for Train data is:", (statistics.mean(npv_array)))
print("Variance of NPV for Train data is:", (statistics.variance(npv_array)))


plt.title("DSC Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,DSC_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("DSC",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('DSC_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("IOU Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,IOU_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("IOU",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('IOU_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Accuracy Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,accu_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Accuracy",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Accuracy_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Sensitivity Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,sens_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Sensitivity",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Sensitivity_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Specificity Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,speci_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Specificity",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Specificity_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Error Rate Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,errorrate_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Error Rate",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Error Rate_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("FNR Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,fnr_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("FNR",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('FNR_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Extra Fraction Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,exfra_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Extra Fraction",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Extra Fraction_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("PPV Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,ppv_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("PPV",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('PPV_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("NPV Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array,npv_array,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("NPV",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('NPV_array.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()