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
import keras.backend as k

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
#IMG_WIDTH = 256
#IMG_HEIGHT = 256
IMG_CHANNELS = 3


#### Add by Mahsa for saving results, found from Faradars website

##### Add by Mahsa for saving results
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

imageset = 'BCC'
backbone = 'UNET'
version = 'v1.0'
model_h5 = 'model-{imageset}-{backbone}-{version}.h5'.format(imageset=imageset, 
                  backbone = backbone, version = version)
model_h5_checkpoint = '{model_h5}.checkpoint'.format(model_h5=model_h5)

earlystopper = EarlyStopping(patience=7, verbose=1)
checkpointer = ModelCheckpoint(model_h5_checkpoint, verbose = 1, save_best_only=True)

#################### Add by Mahsa, found from Internet
from keras.models import model_from_json
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


####################### Propose by Maryam for saving results
# plot_model(model, to_file='modelproposednetworkZJU.png')
# print('training time',str(elapsed))

# Save the weights
model.save_weights('model_weights_test.h5')

# Save the model architecture
with open('model_architecture_test.json', 'w') as f:
    f.write(model.to_json())
#################################################################################

#################################################################################


# 11.1. Summarize History for Loss

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()

matplotlib inline
# 11.1. Summarize History for DSC

plt.plot(results.history['DSC_Evaluator'])
plt.plot(results.history['val_DSC_Evaluator'])
plt.title('DSC')
plt.ylabel('DSC')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()



##############################################################################################
#SAVE the model


from keras.utils import plot_model
#plot_model(model, to_file='model_T2+flaire_UNET_challenge2015_Adam_Cstumloss-1.5-1_DSC.png')
# Save the weights
model.save_weights('model_weights_T2+flaire_UNET_challenge2015_Adam_Cstumloss-1.5-1_DSC.h5')

# Save the model architecture
with open('model_architecture_T2+flaire__UNET_challenge2015_Adam_Cstumloss-1.5-1_DSC.json', 'w') as f:
    f.write(model.to_json())
    
model.save('model_T2+flaire__UNET_challenge2015_Adam_Cstumloss-1.5-1_DSC.h5')

##################################################################################################################################################################  
#Predict the test data and train data
 
from matplotlib import pyplot as plt
idx = random.randint(0, len(X_train))

################### Edit by Mahsa, Change 0.9 to 0.85
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.85)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.85):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


################################################################################# 
#calculate metrics and performance for test data

################### Add by Mahsa 

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
    for i in range(0,preds_test.shape[0]-1-1): ############# edit by Mahsa, add -1
        print(i)
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
    
################ Comment by Mahsa
##counter_array,DSC_array,IOU_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array =parameters(TP_array,TN_array,FP_array,FN_array)

###################### Add new by Mahsa 4 June
i_PPV_Test=[]
for i in range(preds_test_t.shape[0]):        
    if TP_array[i]==FP_array[i]==0:   ############ For PPV
        i_PPV_Test.append(i)
        print(i)


i_DSC_Test=[]
for j in range(preds_test_t.shape[0]):
    if TP_array[j]==FP_array[j]==FN_array[j]==0:    ############# For DSC
        print(j)
        i_DSC_Test.append(j)

###################### For Test Data TP==FP==FN==0 for index 171 
###################### For Test Data TP==FP==0 for indices 86 and 171 
###############For Test data Removing valused of indices 86 and 171 for TP, FP, TN and FN           

####################       
TP_array_Test_edit=[]
for i in range(86):
    TP_array_Test_edit.append(TP_array[i]) 

for i in range(87,171):
    TP_array_Test_edit.append(TP_array[i]) 
    
    
FP_array_Test_edit=[]
for i in range(86):
    FP_array_Test_edit.append(FP_array[i]) 

for i in range(87,171):
    FP_array_Test_edit.append(FP_array[i]) 
  

FN_array_Test_edit=[]
for i in range(86):
    FN_array_Test_edit.append(FN_array[i]) 

for i in range(87,171):
    FN_array_Test_edit.append(FN_array[i]) 
       


TN_array_Test_edit=[]
for i in range(86):
    TN_array_Test_edit.append(TN_array[i]) 

for i in range(87,171):
    TN_array_Test_edit.append(TN_array[i]) 
       
          
##################### Calculate Parameters from New Arrays of TP, FP, TN, FN
counter_array,DSC_array,IOU_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array =parameters(TP_array_Test_edit,TN_array_Test_edit,FP_array_Test_edit,FN_array_Test_edit)
                            

#############

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



import matplotlib.pyplot as plt


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

################### add by Mahsa
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
    for i in range(0,preds_train.shape[0]-1-5):  ############## Add by Mahsa (add -5)
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
    
    
####################### comment by Mahsa    
#counter_array,DSC_array,IOU_array,accu_array,sens_array,speci_array,errorrate_array,fnr_array,exfra_array,ppv_array,npv_array =parameters2(TP1_array,TN1_array,FP1_array,FN1_array)


###################### Add new by Mahsa 4 June
i_PPV=[]
for i in range(preds_train_t.shape[0]):        
    if TP1_array[i]==FP1_array[i]==0:   ############ For PPV
        i_PPV.append(i)
        print(i)


i_DSC=[]
for j in range(preds_train_t.shape[0]):
    if TP1_array[j]==FP1_array[j]==FN1_array[j]==0:    ############# For DSC
        print(j)
        i_DSC.append(j)
            


###################### For Train Data TP==FP==FN==0 not happened,  
############For Train Data TP==FP==0 for indices 878, 2084, 2086, 2087, 2088 and 2089 (6 indices)         
########For Train Data Removing Zeros values of TP, FP, FN, TN for indices 878, 2084, 2086, 2087, 2088 and 2089 (6 indices) 



################# Removing indices i_PPV
TP_array_Train_edit=[]      
for i in range(len(TP1_array)):    
        if not(i in i_PPV):  
            TP_array_Train_edit.append(TP1_array[i])


FP_array_Train_edit=[]      
for i in range(len(FP1_array)):    
        if not(i in i_PPV):  
            FP_array_Train_edit.append(FP1_array[i])
       
TN_array_Train_edit=[]      
for i in range(len(TN1_array)):    
        if not(i in i_PPV):  
            TN_array_Train_edit.append(TN1_array[i])

FN_array_Train_edit=[]      
for i in range(len(FN1_array)):    
        if not(i in i_PPV):  
            FN_array_Train_edit.append(FN1_array[i])


       
############### Add by Mahsa
############### Calculate Parameters from new arrays of TP, FP, TN, FN for train data  
counter_array_Train,DSC_array_Train,IOU_array_Train,accu_array_Train,sens_array_Train,speci_array_Train,errorrate_array_Train,fnr_array_Train,exfra_array_Train,ppv_array_Train,npv_array_Train =parameters2(TP_array_Train_edit,TN_array_Train_edit,FP_array_Train_edit,FN_array_Train_edit)

################## written (edit) by Mahsa, Add (_Train_ at the end of name of parameters
import statistics
print("Mean of DSC for Train data is:", (statistics.mean(DSC_array_Train)))
print("Variance of DSC for Train data is:", (statistics.variance(DSC_array_Train)))

print("Mean of IOU for Train data is:", (statistics.mean(IOU_array_Train)))
print("Variance of IOU for Train data is:", (statistics.variance(IOU_array_Train)))

print("Mean of Accuracy for Train data is:", (statistics.mean(accu_array_Train)))
print("Variance of Accuracy for Train data is:", (statistics.variance(accu_array_Train)))

print("Mean of Sensitivity for Train data is:", (statistics.mean(sens_array_Train)))
print("Variance of Sensitivity for Train data is:", (statistics.variance(sens_array_Train)))

print("Mean of Specificity for Train data is:", (statistics.mean(speci_array_Train)))
print("Variance of Specificity for Train data is:", (statistics.variance(speci_array_Train)))

print("Mean of Error Rate for Train data is:", (statistics.mean(errorrate_array_Train)))
print("Variance of Error Rate for Train data is:", (statistics.variance(errorrate_array_Train)))

print("Mean of FNR for Train data is:", (statistics.mean(fnr_array_Train)))
print("Variance of FNR for Train data is:", (statistics.variance(fnr_array_Train)))

print("Mean of Extra Fraction for Train data is:", (statistics.mean(exfra_array_Train)))
print("Variance of Extra Fraction for Train data is:", (statistics.variance(exfra_array_Train)))

print("Mean of PPV for Train data is:", (statistics.mean(ppv_array_Train)))
print("Variance of PPV for Train data is:", (statistics.variance(ppv_array_Train)))

print("Mean of NPV for Train data is:", (statistics.mean(npv_array_Train)))
print("Variance of NPV for Train data is:", (statistics.variance(npv_array_Train)))


import matplotlib.pyplot as plt

##################### Edit by Mahsa (add _Train at the end of name of Parameters)
plt.title("DSC Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,DSC_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("DSC",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('DSC_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("IOU Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,IOU_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("IOU",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('IOU_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Accuracy Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,accu_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Accuracy",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Accuracy_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Sensitivity Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,sens_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Sensitivity",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Sensitivity_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Specificity Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,speci_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Specificity",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Specificity_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Error Rate Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,errorrate_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Error Rate",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Error Rate_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("FNR Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,fnr_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("FNR",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('FNR_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("Extra Fraction Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,exfra_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("Extra Fraction",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('Extra Fraction_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("PPV Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,ppv_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("PPV",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('PPV_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()


plt.title("NPV Scatter Plot for Train Data", fontsize='16')	#title
plt.scatter( counter_array_Train,npv_array_Train,color='red', marker='o')	#plot the points
plt.xlabel("Number of Image",fontsize='13')	#adds a label in the x axis
plt.ylabel("NPV",fontsize='13')	#adds a label in the y axis
#plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
plt.savefig('NPV_array_Train.png')	#saves the figure in the present directory
plt.grid()	#shows a grid under the plot
plt.show()
####################################################################

# visulaize the performance on some random training samples
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


######################################################   
# visulaize the performance on some random validation samples


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
 


###################################################### 
# visulaize the performance on some random test samples


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
# performance on metrics visualization on some random data

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
#Metric definition

#Accuracy=(TP+TN)/(TP+TN+FN+FP) 
#DSC=(2*TP)/(2*TP+FN+FP)
#Error_rate=(FP+FN)/(TP+TN+FN+FP) 
#Sensitivity=(TP)/(TP+FN)
#Specificity=(TN)/(TN+FP)
#FPR=1-Specificity
#FNR=(FN)/(TN+FN)
#Extra_Fraction=(FP)/(TN+FN)
#PPV=(TP)/(TP+FP)

