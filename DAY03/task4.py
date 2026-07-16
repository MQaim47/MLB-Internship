import cv2
import numpy as np
import matplotlib.pyplot as plt

from task3 import HSV

img = cv2.imread(r"H:\Studies\MLB-Internship\DAY03\image.jpg")

rgb = img[:, :, ::-1]

lower_blue = (100, 50, 50)
upper_blue = (140, 255, 255)

mask = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=np.uint8)

for i in range(HSV.shape[0]):
    for j in range(HSV.shape[1]):

        h, s, v = HSV[i, j]

        if (
            lower_blue[0] <= h <= upper_blue[0]
            and lower_blue[1] <= s <= upper_blue[1]
            and lower_blue[2] <= v <= upper_blue[2]
        ):
            mask[i, j] = 255

result = np.zeros_like(rgb)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):

        if mask[i, j] == 255:
            result[i, j] = rgb[i, j]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Segmented")
plt.axis("off")

plt.tight_layout()
plt.show()