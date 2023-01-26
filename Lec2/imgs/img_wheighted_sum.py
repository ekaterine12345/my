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

result1 = cv.addWeighted(imag1, 0.2, imag2, 0.8, 0)
cv.imshow('result1', result1)

alpha = 0.5
result = alpha * imag1 + (1-alpha)*imag2
result = np.around(result).astype(np.uint8)
cv.imshow('result', result)

cv.waitKey(1000)
