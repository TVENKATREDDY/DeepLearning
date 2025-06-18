import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
from tensorflow.keras.applications.xception import Xception,preprocess_input,decode_predictions
img=load_img(r'C:\Users\91807\VSCodeProjects\PythonPractice\8.DEEP LEARNING\1. TRANSFER LEARNING\images\peacock.jpg',target_size=(299,299))
import time
model=Xception(weights='imagenet')
img_array=img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)
st_time=time.time()
predict=model.predict(img_array)
end_time=time.time()
decoded=decode_predictions(predict,top=3)[0]
for i,(imagenet_id,label,score) in enumerate(decoded):
    print(f'{i+1} : {label} ({score:.2f})')
top_3=np.argmax(predict[0])
print('\n top 3: ',top_3)
inference_time_ms=(end_time-st_time) * 1000.0
print('Inference Time :',inference_time_ms)

model_size_mb=model.count_params()*4/(1024**2)
print(f'Size(MB) :{model_size_mb:.2f} MB')

num_params=model.count_params()
model_depth=len(model.layers)

print(f'Parameters : {num_params}')
print(f'Depth : {model_depth}')
