""" @author: faradars """

'''  Pre_Processing Codes Include:
    
    - One_Hot Encoding
    - Image Re_Sizing
    - Image Augmentation
    - Image Re_Color
    - Image Saving
    - Show Results
'''

# 1. Import Required Modules

import os
import sys
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from skimage.viewer import ImageViewer
from OOP import Pre_Processing_R2g

''' ------------------------------------------------------- '''

# 2. Set Image Path

IMAGE_PATH = 'D:/Python/Breast/Test-X/'
IMAGE_MASK_PATH = 'D:/Python/Breast/mask/'

''' ------------------------------------------------------- '''

# 3. Get Images and Set Some Parameters, One_Hot Encoding Technique

IMG_HEIGHT = 768
IMG_WIDTH = 896
IMG_CHANNELS = 3

IMG_Dataset = next(os.walk(IMAGE_PATH))[2]

Inputs = np.zeros((len(IMG_Dataset), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                  dtype = np.uint8)

#Ground_Truth = np.zeros((len(IMG_Dataset), IMG_HEIGHT, IMG_WIDTH, 2),
#                        dtype = np.bool)
azs = imread(IMAGE_MASK_PATH + 'ytma10_010704_benign1.TIF')
print('Loading Images & Masks, Please Wait')
sys.stdout.flush()
for n, f in tqdm(enumerate(IMG_Dataset), total = len(IMG_Dataset)):
    Images = imread (IMAGE_PATH + f)[:,:,:IMG_CHANNELS]
    Inputs[n] = Images
    
#    Masks = imread(IMAGE_MASK_PATH + f.replace('_ccd.tif', '.TIF'))
#    Masks = np.squeeze(Masks).astype(np.bool)
#    
#    Ground_Truth[n, :, :, 0] = ~Masks
#    Ground_Truth[n, :, :, 1] = Masks
#
#
#print('Loading Images and Masks Completed Successfully!')

''' ------------------------------------------------------- '''

# 4. Define Functions: Re_Size, Augmentation, Re_Color, Im_Saving
# 4.1. Function_1: Pre_Process_Re_Size():

#IMG_HEIGHT_RESIZED = 255
#IMG_WIDTH_RESIZED = 255
#IMG_CHANNELS_RESIZED = 3
#
#def Pre_Process_Re_Size(imgs):
#    
#    img_p = np.zeros((imgs.shape[0], IMG_HEIGHT_RESIZED, IMG_WIDTH_RESIZED
#                      , IMG_CHANNELS_RESIZED), dtype = np.uint8)
#    
#    for i in range(imgs.shape[0]):
#        
#       img_p[i] = resize(imgs[i], (IMG_HEIGHT_RESIZED, IMG_WIDTH_RESIZED,
#               IMG_CHANNELS_RESIZED), preserve_range=True)
#       
#    return img_p
#
#''' ------------------------------------------------------- '''
#
## 4.2. Function_2: Pre_Process_Augmentation():
#
#IMAGE_INITIAL_PATH = '/home/faradars/Desktop/Data/Augmentation/' 
#IMAGE_AUGMENTATED_PATH = '/home/faradars/Desktop/Data/Rotated/' 
#
#Data_Gen = image.ImageDataGenerator(rotation_range=30)
#IMG_AUG = Data_Gen.flow_from_directory(IMAGE_INITIAL_PATH, batch_size=1
#                                       , save_to_dir=IMAGE_AUGMENTATED_PATH,
#                                       save_prefix='Aug', target_size=(768,896))
#
#for i in range(9):
#    IMG_AUG.next()
#    
#def Pre_Process_Augmentation(Path_Images):
#    
#    Images_List = glob.glob(Path_Images)
#    Figure = plt.figure()
#    
#    for i in range (9):
#        Images_A = Image.open(Images_List[i])
#        Sub_Image_Show = Figure.add_subplot(331 + i)
#        Sub_Image_Show.imshow(Images_A)
#    plt.show()
#    return Figure
#    
#Image_Original = Pre_Process_Augmentation(IMAGE_INITIAL_PATH + 'input/*')
#Image_Original.savefig(IMAGE_AUGMENTATED_PATH + '/Original.png', dpi = 200,
#                       papertype = 'a5')
#
#Image_Augmentation = Pre_Process_Augmentation(IMAGE_AUGMENTATED_PATH + '/*')
#Image_Augmentation.savefig(IMAGE_AUGMENTATED_PATH + '/Rotated.png', dpi = 200,
#                       papertype = 'a5') 
#
#
#Image_Resize = Pre_Process_Re_Size(Inputs)

''' ------------------------------------------------------- '''


# 4.3. Function_3: Pre_Process_Re_Color():

Object_Pre_Preocessing = Pre_Processing_R2g(Inputs)
Gray_Scale = Pre_Processing_R2g.R_2_G(Object_Pre_Preocessing)



# 4.4. Function_4: Pre_Process_Im_Saving():

New = 'D:/Python/Breast/'

def Pre_Process_Im_Saving(Path_Images, Path_Output, Tensor):
    
    for i, filename in enumerate(os.listdir(Path_Images)):
        
        imsave(fname='{}{}'.format(Path_Output, filename),
               arr=Tensor[i])
        
        print('{}: {}'.format(i, filename))
    
Pre_Process_Im_Saving(IMAGE_PATH, New, Gray_Scale)

''' ------------------------------------------------------- '''
#
## 5. Show Results
#
#ix = random.randint(0, len(Inputs))
#
#img = Inputs[ix]
#mask = Ground_Truth [ix]
#resized = Image_Resize [ix]
#gray = Gray_Scale[ix]
#
#print('Input Image')
#imshow(img)
#plt.show()
#
#print('Mask')
#imshow(mask[:,:,1])
#plt.show()
#
#print('Resized Image')
#imshow(resized)
#plt.show()
#
#print('GrayScale Image')
#imshow(gray)
#plt.show()
#
#
#image_viewer = ImageViewer(gray)
#image_viewer.show()












