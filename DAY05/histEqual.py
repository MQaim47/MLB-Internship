import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg",0)

hist = np.zeros(256, dtype=int)

for row in img:
    for pixel in row:
        hist[pixel] += 1

total_pixels = img.shape[0] * img.shape[1]

pdf= hist / total_pixels

cdf = np.cumsum(pdf)

mapping = np.round(255 * cdf).astype(np.uint8)

equalized_img = np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        equalized_img[i,j] = mapping[img[i,j]]
        
cv2.imwrite("equalized_image.jpg", equalized_img)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image")
plt.show()
plt.savefig("equalized_comparison.png")