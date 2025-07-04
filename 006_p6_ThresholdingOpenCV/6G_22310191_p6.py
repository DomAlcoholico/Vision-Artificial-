import cv2
import numpy as np

img = cv2.imread('bookpage.jpg')
img = cv2.resize(img, (500, 370))
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
