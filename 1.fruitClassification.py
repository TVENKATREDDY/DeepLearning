import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
image_height,image_width=32,22
batch_size=20
train_ds=image_dataset_from_directory(r'C:\Venkat\Python\Practice_Material\DEEP LEARNING(10th Jan)\fruits\train',
                                      image_size=(image_height,image_width),
                                      batch_size=batch_size
                                      )
val_ds=image_dataset_from_directory(r'C:\Venkat\Python\Practice_Material\DEEP LEARNING(10th Jan)\fruits\validation',
                                    image_size=(image_height,image_width),
                                    batch_size=batch_size
                                    )

test_ds=image_dataset_from_directory(r'C:\Venkat\Python\Practice_Material\DEEP LEARNING(10th Jan)\fruits\test',
                                     image_size=(image_height,image_width),
                                     batch_size=batch_size
                                     )


class_names=['apple','banana','orange']
plt.figure(figsize=(10,10))
for images,lables in train_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[lables[i]])
        plt.axis('off')
        