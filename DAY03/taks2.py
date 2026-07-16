import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.jpg")
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Unable to load image from: {image_path}")

img = img[:, :, ::-1]

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

# merged = cv2.merge([R,G,B])

plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(R,cmap="gray")
plt.title("Red Channel")

plt.subplot(1,3,2)
plt.imshow(G,cmap="gray")
plt.title("Green Channel")

plt.subplot(1,3,3)
plt.imshow(B,cmap="gray")
plt.title("Blue Channel")
plt.show()

merged = np.zeros_like(img)
merged[:, :, 0] = R
merged[:, :, 1] = G
merged[:, :, 2] = B

plt.figure(figsize=(8, 4))
plt.imshow(merged)
plt.title("Merged Image")
plt.axis("off")
plt.show()
