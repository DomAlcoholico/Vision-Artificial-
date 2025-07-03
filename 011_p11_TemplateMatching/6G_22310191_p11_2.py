
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer imágenes a color (quitamos el '0')
img1 = cv2.imread('mclaren.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('mclaren.jpg', cv2.IMREAD_COLOR)

# Convertir a escala de grises solo para ORB
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

# Detectar y computar en escala de grises (más confiable)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar coincidencias sobre imágenes a color
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# Mostrar la imagen en colores con matplotlib (convertir de BGR a RGB)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
