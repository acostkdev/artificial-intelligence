import cv2 as cv
import numpy as np

img = cv.imread('imagen.jpg', 1)
img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('BGR', img)
cv.imshow('RGB', img2)

zero = np.zeros(img.shape[:2], dtype='uint8')
r, g, b = cv.split(img)

cv.imshow('GRB', cv.merge([g, r, b]))
cv.imshow('R', cv.merge([zero, zero, r]))
cv.imshow('G', cv.merge([zero, g, zero]))
cv.imshow('B', cv.merge([b, zero, zero]))

cv.waitKey(0)
cv.destroyAllWindows()
