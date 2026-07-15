import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

L,A,B = cv2.split(lab)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(L,cmap='gray')
plt.title("L Channel")

plt.subplot(1,3,2)
plt.imshow(A,cmap='gray')
plt.title("A Channel")

plt.subplot(1,3,3)
plt.imshow(B,cmap='gray')
plt.title("B Channel")

plt.show()