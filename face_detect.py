# importing librarys
import cv2 
import numpy as np

import face_recognition  

img = cv2.imread(r'Resources\Photos\1.jpg')    
cv2.imshow('Group', img) 

# Convert into grayscale  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Group', gray)

Cam = face_recognition.load_image_file(r'Resources\Faces\Cam.jpg')   
Cam = cv2.cvtColor(Cam, cv2.COLOR_BGR2RGB) 

faceLoc = face_recognition.face_locations(Cam)[0] 
encodeCam = face_recognition.face_encodings(Cam)[0] 
cv2.rectangle(Cam,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)    

# Load mô hình nhận diện khuôn mặt 
haar_cascade = cv2.CascadeClassifier('haar_face.xml')

# Detect faces  
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)   

print(f'Number of faces found = {len(faces_rect)}')

# Draw rectangle around the faces  
for (x,y,w,h) in faces_rect:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv2.putText(Cam,f'Cam Tran',(70,120), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) 
cv2.putText(img,f'Cam Tran',(330,150), cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)

# Display the output  
cv2.imshow('Cam Tran', Cam)  
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)


# To capture video from existing video.   
cap = cv2.VideoCapture(r'Resources\Videos\1.mp4')     

while True:  
    # Read the frame  
    _, img = cap.read()  
  
    # Convert to grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = haar_cascade.detectMultiScale(gray, 1.2, 4)     
  
    # Draw the rectangle around each face  
    for (x, y, w, h) in faces:  
       cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)   
      
       cv2.putText(img,f'Cam Tran',(330,50), cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)

    # Display  
    cv2.imshow('Video', img)  
  
    # Stop if escape key is pressed     
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
          
# Release the VideoCapture object  
cap.release()  

