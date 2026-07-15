import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(rgb)
plt.title("RGB")

plt.subplot(2,3,2)
plt.imshow(hsv)
plt.title("HSV")

plt.subplot(2,3,3)
plt.imshow(hsl)
plt.title("HSL")

plt.subplot(2,3,4)
plt.imshow(lab)
plt.title("LAB")

plt.subplot(2,3,5)
plt.imshow(gray,cmap='gray')
plt.title("Grayscale")

plt.show()