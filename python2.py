import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Reading the Image
img1 = cv.imread('trial1.jpg')
cv.imshow('Original Image 1',img1)
img2 = cv.imread('trial2.jpg')
cv.imshow('Original Image 2',img2)

# Converting BGR to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray1', gray1)
#cv.imshow('Gray2', gray2)

blur1 = cv.GaussianBlur(gray1, (7,7), cv.BORDER_DEFAULT)
blur2 = cv.GaussianBlur(gray2, (3,3), cv.BORDER_DEFAULT)
#cv.imshow('Blur1', blur1)
#cv.imshow('Blur2', blur2)

#converting to binary (Only black and white)
ret,thresh1 = cv.threshold(blur1,100,155,cv.THRESH_BINARY)
cv.imshow('Output Image 1',thresh1)
ret,thresh2 = cv.threshold(blur2,150,255,cv.THRESH_BINARY)
cv.imshow('Output Image 2',thresh2)

# Edge Cascade
canny1 = cv.Canny(blur1, 130, 130)
#cv.imshow('Canny Edges 1', canny1)
canny2 = cv.Canny(blur2, 150, 150)
#cv.imshow('Canny Edges 2', canny2)

# Dilation (thickenning the boreders)
dilated1 = cv.dilate(canny1, (3,3), iterations=7)
#cv.imshow('Dilated 1', dilated1)
dilated2 = cv.dilate(canny2, (7,7), iterations=3)
#cv.imshow('Dilated 2', dilated2)

cv.waitKey(0)
cv.destroyAllWindows()

# Viewing using matplotlib
plt.imshow(thresh1,cmap='grey')
plt.show()
plt.imshow(thresh2,cmap='grey')
plt.show()