import numpy as np
from tensorflow.keras.preprocessing.image import load_img,array_to_img,img_to_array
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
import tensorflow.keras as keras
import time
model=VGG16(weights='imagenet')
img=load_img(r'C:\Venkat\Venkat_Photos\kittu.jpeg',target_size=(224,224))
img_array=img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)
st_time=time.time()
predictions=model.predict(img_array)
end_time=time.time()
decoded=decode_predictions(predictions,top=3)[0]
print('Predictions:')
for i,(imagenet_id,lable,score) in enumerate(decoded):
    print(f'{i +1} : {lable} ({score:.2f})')
       
top_class_index=np.argmax(predictions[0])
print(f'\n top prediction class index:,{top_class_index}')
inf_time=(end_time-st_time)*1000.0
num_params=model.count_params()
msize=model.count_params()*4/(1024**2)
depth=len(model.layers)
print(f'Inference Time :{inf_time}')
print(f'Model Size :{msize:.2f} MB')
print(f'Number of Parameters :{num_params}')
print(f'Depth :{depth}')    
