import pandas as pd
import numpy as np  
import tensorflow as tf
cutsData=pd.read_csv(r'C:\Venkat\Python\Practice_Material\DEEP LEARNING\17th Jan - ANN THEORY, Installation\9th,10th,17th - ANN THEORY, Installation\Practicle - CPU\ANN_ 1st\Churn_Modelling.csv')
x=cutsData.iloc[:,3:-1].values
y=cutsData.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=ct.fit_transform(x)
#Feature Scalin
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)

ann=tf.keras.models.Sequential()

#Adding Input Layer and 1st hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#adding 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=5,activation='relu'))
ann.add(tf.keras.layers.Dense(units=4,activation='relu'))

#Adding output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Training ANN
#Compiling ANN
#ann.compile(optimizer='SGD',loss='binary_crossentropy',metrics=['accuracy'])
#ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics=['accuracy'])
#ann.compile(optimizer='adagrad',loss='binary_crossentropy',metrics=['accuracy'])
#ann.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
#ann.compile(optimizer='Nadam',loss='binary_crossentropy',metrics=['accuracy'])
#ann.fit(xtrain,ytrain,batch_size=32,epochs=10)

ypred=ann.predict(xtest)
ypred=(ypred > 0.5)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
print(cm)
print(ac)


