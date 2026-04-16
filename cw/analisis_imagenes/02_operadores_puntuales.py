import cv2 as cv
import numpy as np

img = cv.imread('imagen.jpg', 0)
print(img.shape, img.size)
w = img.shape[0]
h = img.shape[1]

img2 = img.copy()
cv.imshow('original', img2)

for x in range(w):
    for y in range(h):
        if img[x, y] > 50:
            img[x, y] = 255
        else:
            img[x, y] = 0

cv.imshow('umbralizada', img)
cv.waitKey(0)
cv.destroyAllWindows()
