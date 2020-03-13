#import libraries
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense,Conv1D,MaxPooling1D
from keras.models import Model


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

tf.compat.v1.enable_eager_execution()

#load the dataset
nsynth_train = tfds.load(name="nsynth", split="train")
nsynth_test = tfds.load(name="nsynth", split="test")
nsynth_validation = tfds.load(name="nsynth",split="valid")
assert isinstance(nsynth_train, tf.data.Dataset)
assert isinstance(nsynth_test, tf.data.Dataset)
assert isinstance(nsynth_validation, tf.data.Dataset)

#set the batch size
nsynth_train = nsynth_train.repeat().shuffle(1024).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
nsynth_test = nsynth_test.repeat().shuffle(1024).batch(32)

#aux lists
X = []
Y = []
XX = []
YY = []

#extraction of the features for the train set
for nsynth_example in nsynth_train.take(300):  # Only take a single example
      gc.collect()
      pitch,label,family,idx,audio,source = nsynth_example["pitch"].numpy(), nsynth_example["instrument"]["label"],nsynth_example["instrument"]["family"].numpy(),nsynth_example["id"],nsynth_example["audio"].numpy(),nsynth_example["instrument"]["source"].numpy()
      for i in range(len(pitch)):
         if (pitch[i]==60):
             X.append(audio[i])
             Y.append(family[i])
         
      

#extraction of the features for the test set
for nsynth_example in nsynth_test.take(100):  # Only take a single example
      gc.collect()
      pitch,label,family,idx,audio,source = nsynth_example["pitch"].numpy(), nsynth_example["instrument"]["label"],nsynth_example["instrument"]["family"].numpy(),nsynth_example["id"],nsynth_example["audio"].numpy(),nsynth_example["instrument"]["source"].numpy()
      for i in range(len(pitch)):
         if (pitch[i]<60 and pitch[i]>10):
             print(source[i])
             XX.append(audio[i])
             YY.append(family[i])

#reshape of the lists in order to have a valid input for the neural network

Y = tf.stack(Y)
Y_new = np.array(Y)
X = tf.stack(X)
X_new = np.array(X) 
X_new = X_new.reshape(len(X_new),64000,1).astype('float32')
Y_new = Y_new.reshape(len(Y_new),1).astype('int64')
XX = tf.stack(XX)
YY = tf.stack(YY)
XX_new = np.array(XX)
YY_new = np.array(YY)
XX_new = XX_new.reshape(len(XX_new),64000,1).astype('float32')
YY_new = YY_new.reshape(len(YY_new),1).astype('int64')


#Convolutional neural network model

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64,9,activation='relu', input_shape=(64000,1)),
    tf.keras.layers.MaxPooling1D(9),
   
   #Uncomment to add convolutional layers  
   # tf.keras.layers.Conv1D(32,32,activation='relu'),
   # tf.keras.layers.MaxPooling1D(32),
   # tf.keras.layers.Conv1D(16,64,activation='relu'),
   # tf.keras.layers.MaxPooling1D(32),
   # tf.keras.layers.Conv1D(8,128,activation='relu'),
   # tf.keras.layers.Conv1D(4,256,activation='relu'),
    
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(Y_new),activation=tf.nn.softmax)
])

#Model summary
model.summary()
#Compiling, training and test the model
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_new,Y_new,epochs=20) 
model.evaluate(XX_new,YY_new,verbose=2,batch_size = 32)

