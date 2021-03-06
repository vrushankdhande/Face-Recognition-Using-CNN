# -*- coding: utf-8 -*-
"""CNN_Training_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m1Iq2XucjLhVwo878gCWIwDJhhIPFz4c
"""

#print(confusion matrix)
predictions = model.predict(X_test)
y_pred=(predictions > 0.5)
matrix = metrics.confusion_matrix(y_test_arr.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

#CNN code 
from keras.models import Sequential   
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import activations
import tensorflow as tf
import keras
number_classes=7

model=Sequential()
model.add(Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(130,130,3)))    #used to name the model with the training info
model.add(MaxPooling2D(pool_size=(16, 16)))
model.add(Dropout(0.2)) 

model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))    #used to name the model with the training info
model.add(MaxPooling2D(pool_size=(4, 4)))                           #add more layer copy this past 
model.add(Dropout(0.2)) 

model.add(Flatten())                                                #tthis are input layers
model.add(Dense(16, activation=tf.nn.relu))                         #past some parametors for layers and have a  activation fuction which help the nuron to fire
model.add(Dense(number_classes))
model.add(Dense(number_classes, activation=tf.nn.softmax))             # SM for probability
                                                                    #this the architecture for train the model is above.....
model.summary()                                                     # check model structure and the number of parameters
                                                                    
model.compile(optimizer=keras.optimizers.Adam(),  #default go to optimizer is used adam which very .
              loss=keras.losses.categorical_crossentropy,  #used too categorical the data 
              metrics=["accuracy"])   

y_p=model.fit(X_train,y_train_arr,batch_size=32,epochs = 10,steps_per_epoch =50,validation_data=(X_val, y_val_arr),validation_steps = 16)

#result
results = model.evaluate(X_test, y_test_arr, batch_size=32)
print("test loss, test acc:", results)

#matrix confusion
import sklearn
import sklearn.metrics 
from sklearn.metrics import confusion_matrix
predictions = model.predict(X_test)
y_pred=(predictions > 0.5)
matrix = confusion_matrix(y_test_arr.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

#save and load the model
model.save('/content/gdrive/MyDrive/face_recognition/Faces/add/save/gg/model_11'+ '.h5')

loaded_model = tf.keras.models.load_model('/content/gdrive/MyDrive/face_recognition/Faces/save/gg/model_6/')