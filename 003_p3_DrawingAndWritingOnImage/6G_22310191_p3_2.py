import numpy as np
import cv2
import matplotlib.pyplot as plt

# Leer y verificar la imagen
img = cv2.imread('storm.jpg', cv2.IMREAD_COLOR)
if img is None:
    print(" Error: No se pudo cargar la imagen.")
    exit()

# Redimensionar la imagen base
img = cv2.resize(img, (350, 400))

# Dibujos sobre la imagen
cv2.line(img, (0, 0), (150, 150), (255, 255, 255), 15)
cv2.rectangle(img, (40, 50), (300, 250), (0, 255, 0), 5)
cv2.circle(img, (100, 63), 55, (0, 0, 255), -1)

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
cv2.polylines(img, [pts], True, (0, 255, 255), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'ImDominickOpenCVtuts', (0, 330), font, 1, (200, 255, 255), 2, cv2.LINE_AA)

# Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ecualizar imagen en escala de grises
equalized = cv2.equalizeHist(gray)

# Calcular histogramas
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# Redimensionar todas las imágenes a 2/3 de su tamaño
def resize_2_3(image):
    return cv2.resize(image, (0, 0), fx=2/3, fy=2/3)

img_small = resize_2_3(img)
gray_small = resize_2_3(gray)
equalized_small = resize_2_3(equalized)

# Mostrar imágenes con OpenCV redimensionadas
cv2.imshow('Imagen con Dibujos (2/3)', img_small)
cv2.imshow('Escala de Grises (2/3)', gray_small)
cv2.imshow('Imagen Ecualizada (2/3)', equalized_small)

# Mostrar histogramas con matplotlib más pequeños
plt.figure(figsize=(6.7, 2.7))  # 2/3 del tamaño original (10x4)

plt.subplot(1, 2, 1)
plt.title('Histograma Original (Grises)')
plt.plot(hist_gray, color='gray')
plt.xlim([0, 256])

plt.subplot(1, 2, 2)
plt.title('Histograma Ecualizado')
plt.plot(hist_eq, color='gray')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
