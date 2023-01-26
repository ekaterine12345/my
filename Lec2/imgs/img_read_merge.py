import  numpy as np
import cv2 as cv
imag1 = cv.imread('logos.png')
shape1 = imag1.shape  # The dimensions of a given image like the height of the image, width of the image and number of
# channels in the image are called the shape of the image. The shape of the image is stored in numpy. ndarray.

# print(shape1[0], shape1[1])
imag2 = cv.imread('arduino.png')

imag2 = cv.resize(imag2, (shape1[1], shape1[0]))

cv.imshow('logos', imag1)
cv.imshow('arduino', imag2)

print(imag1.shape)  # (348, 625, 3)
print(imag2.shape)  # (348, 630, 3)

result = imag1 + imag2
cv.imshow('result', result)
cv.waitKey(10000)
