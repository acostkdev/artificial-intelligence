import cv2
import numpy as np

imagen_color = cv2.imread('imagen.jpg')
plantilla_color = cv2.imread('plantilla.jpg')

imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
plantilla_gris = cv2.cvtColor(plantilla_color, cv2.COLOR_BGR2GRAY)

altura, ancho = plantilla_gris.shape

resultado = cv2.matchTemplate(imagen_gris, plantilla_gris, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)

top_left = max_loc
bottom_right = (top_left[0] + ancho, top_left[1] + altura)

cv2.rectangle(imagen_color, top_left, bottom_right, (0, 255, 0), 2)

cv2.imshow('resultado', imagen_color)
cv2.imshow('resultado2', resultado)

cv2.waitKey()
cv2.destroyAllWindows()
