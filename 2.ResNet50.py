from tensorflow.keras.utils import image_dataset_from_directory,array_to_img,img_to_array,load_img
#from keras.utils import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
import numpy as np
import time

keras.applications.ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
    )
#img=load_img(r'C:\Venkat\Venkat_Photos\08-06-2022_M31_Photos\20210328_070911.jpg')
model=ResNet50(weights='imagenet')
img_path=r'C:\Venkat\Venkat_Photos\08-06-2022_M31_Photos\20211115_021124.jpg'
img=image.load_img(img_path,target_size=(224,224))
img_array=image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)
st_time=time.time()
predict=model.predict(img_array)
end_time=time.time()
decoded_predictions=decode_predictions(predict,top=3)[0]
print('Predictions :')
for i,(imagenet_id,label,score) in enumerate(decoded_predictions):
    print(f'{i + 1}: {label} ({score:.2f})')
    
top_class_index=np.argmax(predict[0])
print(f'\n Top Prediction Class Index: {top_class_index}')
inf_time=(end_time-st_time)*1000.0
num_params=model.count_params()
msize=model.count_params()*4/(1024**2)
depth=len(model.layers)
print(f'Inference Time :{inf_time}')
print(f'Model Size :{msize:.2f} MB')
print(f'Number of Parameters :{num_params}')
print(f'Depth :{depth}')

