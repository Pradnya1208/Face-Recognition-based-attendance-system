import cv2
import numpy as np
import face_recognition

IMAGE = 'Einstein-Test_2.jpg'

import os
PATH = 'Images'
images = []
classNames = []
myList = os.listdir(PATH)


for im in myList:
    curImg = cv2.imread(f'{PATH}/{im}')
    images.append(curImg)
    classNames.append(os.path.splitext(im)[0])
    
def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList        

encodelist_known = findEncodings(images)

img =  cv2.imread(IMAGE)

image_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
facesCurFrame  = face_recognition.face_locations(image_test)
encodeCurFrame = face_recognition.face_encodings(image_test, facesCurFrame)
    
for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
    match_faces = face_recognition.compare_faces(encodelist_known, encodeFace)
    faceDis = face_recognition.face_distance(encodelist_known, encodeFace)
    #print(match_faces)
    matchIndex = np.argmin(faceDis)
        
    if match_faces[matchIndex]:
        name = classNames[matchIndex].upper()
        face_loc = faceLoc
        
cv2.rectangle(img, (face_loc[3],face_loc[0]), (face_loc[1],face_loc[2]),(255,0,255),2) # 0- top, 1- right, 2 - bottom, 3 -left
cv2.putText(img, classNames[matchIndex], (face_loc[3],face_loc[2] +20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
cv2.imshow('TEST IMAGE', img)
cv2.imshow(classNames[matchIndex].upper(), images[matchIndex])
cv2.waitKey(0)