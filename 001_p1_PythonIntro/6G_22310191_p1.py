import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('storm.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (550, 600))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
