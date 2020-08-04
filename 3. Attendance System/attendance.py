import cv2
from cv2 import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

PATH = 'Images'
images = []
classNames = []
myList = os.listdir(PATH)

# Create a list of images and annotations
for im in myList:
    curImg = cv2.imread(f'{PATH}/{im}')
    images.append(curImg)
    classNames.append(os.path.splitext(im)[0])
    
# Encode the images in a list    
def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList        

def recordAttendance(name):
    with open('attendance.csv', 'r+') as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateStr}')
            
            
        



encodelist_known = findEncodings(images)

# Webcam capture
cap = cv2.VideoCapture(0)


while True:
    # Encoding webcam capture
    match, img =  cap.read()
    image_resize = cv2.resize(img, (0,0), None, 0.25, 0.25)
    image_resize = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame  = face_recognition.face_locations(image_resize)
    encodeCurFrame = face_recognition.face_encodings(image_resize, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        match_faces = face_recognition.compare_faces(encodelist_known, encodeFace)
        faceDis = face_recognition.face_distance(encodelist_known, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if match_faces[matchIndex]:
            name = classNames[matchIndex].upper()
            print (name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            recordAttendance(name)            
            
    cv2.imshow('webcam', img)
    cv2.waitKey(1)