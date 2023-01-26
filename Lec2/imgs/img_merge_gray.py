import numpy as np
import cv2 as cv

imag1 = cv.imread('logos.png')
imag1 = cv.cvtColor(imag1, cv.COLOR_BGR2GRAY)
shape1 = imag1.shape
print(shape1)

imag2 = cv.imread('arduino.png')
imag2 = cv.cvtColor(imag2, cv.COLOR_BGR2GRAY)
shape2 = imag2.shape
print(shape2)

imag2 = cv.resize(imag2, (shape1[1], shape1[0]))

cv.imshow('logos', imag1)

cv.imshow('arduino', imag2)

result = imag1 + imag2
cv.imshow('result', result)
cv.waitKey(0)
