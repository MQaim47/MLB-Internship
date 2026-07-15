import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread("image.jpg")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_bgr)
plt.title("BGR Image")

plt.subplot(1,2,2)
plt.imshow(img_rgb)
plt.title("RGB Image")

plt.show()