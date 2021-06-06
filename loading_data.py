# -*- coding: utf-8 -*-
"""Loading_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TO-qgednChKODuCB3srv7tJNIWPCUlGY
"""

# the program will extract the faces from the images /// for only one folder
# for more than one folder in the group

# the program will extract the faces from the images
import cv2
import sys
import os
import numpy as np
from PIL import Image as pil_image
from google.colab.patches import cv2_imshow

count=0

harcascadePath = '/content/gdrive/MyDrive/face_recognition/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(harcascadePath)
th = '/content/gdrive/MyDrive/face_recognition/Faces/bro/'
imagePaths = [os.path.join(th,f) for f in os.listdir(th)]
new = '/content/gdrive/MyDrive/dd/'
face_arrr = []
label_arrr = []

for temp_label in imagePaths:
  try:
      for path in os.listdir(temp_label):
          image = cv2.imread(temp_label+'/'+path)
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          faces = detector.detectMultiScale(gray, 1.3, 5)
          if faces is not ():
              for (x,y,w,h) in faces:
                  #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1,1)
                  #label code
                  label=os.path.basename(os.path.dirname(temp_label+'/'+path))
                  print(label)
                  face = image[y-50:y+h+50,x-50:x+w+50]
                  face_resize=cv2.resize(face,(130,130))
                  count += 1
                  print(path)
                  print(label)
                  cv2_imshow(face_resize)
                  # cv2.imwrite(new+str(count)+ 'face' + ".jpg",face)
                  face_arrr.append(face_resize)
                  label_arrr.append(label)  
  except:
      print("exception occured")
      continue    

# cv2_imshow(face)

import os
import cv2
from google.colab.patches import cv2_imshow
face_arrr = []
label_arrr = []
unknown = "/content/gdrive/MyDrive/face_recognition/Faces/add/Unknown/"
for label_unknown in os.listdir(unknown):
    unknowns_path = os.path.join(unknown, label_unknown)
    image_unknown = cv2.imread(unknowns_path)
    unknown_labels=os.path.basename(os.path.dirname(unknown+'/'+label_unknown))
    unknown_img_resize =cv2.resize(image_unknown,(130,130))
    print(unknown_labels)
    cv2_imshow(unknown_img_resize)
    face_arrr.append(unknown_img_resize)
    label_arrr.append(unknown_labels) 

#check the all the classes

classes = set(label_arrr)
number_classes = list(classes)
print(len(number_classes))
number_classes=len(number_classes)
classes
label_arrr.count('Manav')