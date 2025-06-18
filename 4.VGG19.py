import numpy as np
from tensorflow.keras.preprocessing.image import load_img,array_to_img,img_to_array
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions
import tensorflow.keras as keras
model=VGG19(weights='imagenet')
img=load_img(r'C:\Venkat\Venkat_Photos\peacock.jpg',target_size=(224,224))
img_array=img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)
predictions=model.predict(img_array)
decoded=decode_predictions(predictions,top=3)[0]
for i,(imagenet_id,label,score) in enumerate(decoded):
    print(f'{i+1} : {label} ({score:.2f})')
top_3=np.argmax(predictions[0])
print('\n top 3 index :',top_3)
