import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg",0)

equalized = cv2.equalizeHist(img)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)  
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Equalized Image")

plt.subplot(1,3,3)
plt.imshow(clahe_img,cmap='gray')
plt.title("CLAHE")

plt.show()