# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:35:44 2021

@author: vrush
"""

from PIL import Image
#from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
#tf.keras.models.load_model(model_path)

from keras.preprocessing import image
model = load_model('model_7.h5')
#tf.keras.models.load_model(model_path)
# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cropped_face = img[y:y+h, x:x+w]
        #print(cropped_face)

    return cropped_face

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (130, 130))
        im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        #print(img_array)
        pred = model.predict(img_array)
        print(pred)
                     
        names="None matching"
        ff = 'Unknown'
        #gg =
        ##gd=
        #gr=
        #ge=
        if(pred[0][6]>0.004):
            name=['Vrushank', 'Manav', 'Unknown']
            fg=list(pred)
            names=name[fg.index(max(fg))]
            print(names)
            cv2.putText(frame,names,(50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        else :
            (pred[0][6]>0.0015)
            cv2.putText(frame,ff,(50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        
    else:
        cv2.putText(frame,"FACE NOT FOUND", (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()