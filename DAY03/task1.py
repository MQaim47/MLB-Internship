import cv2
import matplotlib.pyplot as plt
import numpy as np

img_bgr = cv2.imread("H:\\Studies\\MLB-Internship\\DAY03\\image.jpg")

img_rgb=img_bgr[:,:,::-1]
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_bgr)
plt.title("BGR Image")

plt.subplot(1,2,2)
plt.imshow(img_rgb)
plt.title("RGB Image")

plt.show()