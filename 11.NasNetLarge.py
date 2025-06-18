import numpy as np
import time
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
from tensorflow.keras.applications.nasnet import NASNetLarge,preprocess_input,decode_predictions
model=NASNetLarge(weights='imagenet')
img=load_img(r'C:\Users\91807\VSCodeProjects\PythonPractice\8.DEEP LEARNING\1. TRANSFER LEARNING\images\peacock.jpg',target_size=(331,331))
img_array=img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)
st_time=time.time()
predict=model.predict(img_array)
end_time=time.time()
decoded=decode_predictions(predict,top=6)[0]
for i,(imagenet_id,label,score) in enumerate(decoded):
    print(f'{i+1} : {label} ({score:.2f})')
top_6=np.argmax(predict[0])
inf_time=(end_time-st_time)*1000.0
num_params=model.count_params()
msize=model.count_params()*4/(1024**2)
depth=len(model.layers)
print(f'Inference Time :{inf_time}')
print(f'Model Size :{msize:.2f} MB')
print(f'Number of Parameters :{num_params}')
print(f'Depth :{depth}')