import numpy as np
from tensorflow.keras.preprocessing.image import load_img,array_to_img,img_to_array
from tensorflow.keras.applications.resnet_v2 import ResNet50V2,preprocess_input,decode_predictions
import tensorflow.keras as keras
model=ResNet50V2(weights='imagenet')
img=load_img(r'C:\Users\91807\VSCodeProjects\PythonPractice\8.DEEP LEARNING\1. TRANSFER LEARNING\images\peacock.jpg',target_size=(224,224))
img_array=img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)
#st_time=time.time()
predict=model.predict(img_array)
#end_time=time.time()
decoded=decode_predictions(predict,top=3)[0]
for i,(imagenet_id,label,score) in enumerate(decoded):
    print(f'{i+1} : {label} ({score:.2f})')
top_3=np.argmax(predict[0])
print('\n top index :',top_3)
#time_taken=(end_time-st_time)*1000.0
#print('Time Taken :',time_taken)
model_size=model.count_params() * 4 /(1024**2)
print(f'Model Size(MB) :',model_size)
num_parameters=model.count_params()
model_depth=len(model.layers)
print(f'Parameters: {num_parameters}')
print(f'Depth :{model_depth}')