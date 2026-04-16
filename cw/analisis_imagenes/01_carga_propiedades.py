import cv2 as cv
import numpy as np

img = cv.imread("imagen.jpg", 1)
print("shape:", img.shape)

img2 = np.zeros(img.shape[:2], np.uint8)
b, g, r = cv.split(img)

b1 = cv.merge([b, img2, img2])
g1 = cv.merge([img2, g, img2])
r1 = cv.merge([img2, img2, r])
res1 = cv.merge([g, r, b])

cv.imshow('res1', res1)
cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)
cv.imshow('b1', b1)
cv.imshow('g1', g1)
cv.imshow('r1', r1)
cv.imshow('img2', img2)
cv.imshow('imagen', img)

print("w, h, c:", img.shape)

cv.waitKey(0)
cv.destroyAllWindows()
