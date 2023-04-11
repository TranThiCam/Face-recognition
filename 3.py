import cv2 
import numpy as np

#img = cv2.imread(r'Resources\Photos\1.jpg')
img = cv2.imread(r'Resources\Faces\Cam.jpg')

cv2.imshow('Photo', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

#Gaussian Blue 
gauss = cv2.GaussianBlur(img, (3,3), 0)
cv2.imshow('Gaussian Blur', gauss) 

# Bilateral
bilateral = cv2.bilateralFilter(img, 10, 35, 25)
cv2.imshow('Bilateral', bilateral)

# Laplacian
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow('Laplacian', lap)

# Sobel 
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
combined_sobel = cv2.bitwise_or(sobelx, sobely)


cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Combined Sobel', combined_sobel)

canny = cv2.Canny(gray, 150, 175)
cv2.imshow('Canny', canny)

cv2.waitKey(0) 