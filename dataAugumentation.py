from tensorflow.keras.utils import image_dataset_from_directory
from keras.utils import array_to_img,img_to_array,load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import tensorflow.keras as keras


datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

img=load_img(r'C:\Venkat\Venkat_Photos\08-06-2022_M31_Photos\20210328_070911.jpg')
#print(img)

x=img_to_array(img) # this numpy array with shape 3262,3001,3
#print('shape :',x.shape)
x=x.reshape((1,)+x.shape) # this is numpy array 1,3262,3001,3
print('x shape :',x.shape)
i=0

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
for batch in datagen.flow(x,batch_size=1,save_to_dir=r'C:\Venkat\Venkat_Photos\08-06-2022_M31_Photos\Augument',save_prefix='tree',save_format='jpg'):
    i+=1
    if i> 2:
        break

