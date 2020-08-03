import cv2
import numpy as np
import face_recognition
IMAGE = 'Images/Elon-Musk.jpg'
IMAGE_TEST = 'Images/Elon-Test.jpg'

imgElon = face_recognition.load_image_file(IMAGE)
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElon_test = face_recognition.load_image_file(IMAGE_TEST)
imgElon_test = cv2.cvtColor(imgElon_test, cv2.COLOR_BGR2RGB)

face_location = face_recognition.face_locations(imgElon)[0]
encode_elon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (face_location[3],face_location[0]),
              (face_location[1],face_location[2]),(255,0,255),2) # top, right, bottom, left


faceLoc_test = face_recognition.face_locations(imgElon_test)[0]
encode_test = face_recognition.face_encodings(imgElon_test)[0]
cv2.rectangle(imgElon_test,(faceLoc_test[3],faceLoc_test[0]),
              (faceLoc_test[1],faceLoc_test[2]),(255,0,255),2) # top, right, bottom, left

results = face_recognition.compare_faces([encode_elon], encode_test)
faceDis = face_recognition.face_distance([encode_elon], encode_test)
print (results, faceDis)

if results:
    result = ' Match found '
else:
    result = ' No Match '
    
cv2.putText(imgElon, 'Elon Musk' , (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2 )
cv2.putText(imgElon_test, f'{result} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2 )

cv2.imshow('Elon-Musk', imgElon)
cv2.imshow('Elon-test', imgElon_test)
cv2.waitKey(0)

