import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg")

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


noise_rgb = np.random.normal(
    0,
    25,
    img.shape
)

noisy_rgb = img.astype(np.float32) + noise_rgb

noisy_rgb = np.clip(
    noisy_rgb,
    0,
    255
).astype(np.uint8)



noise_gray = np.random.normal(
    0,
    25,
    gray.shape
)

noisy_gray = gray.astype(np.float32) + noise_gray

noisy_gray = np.clip(
    noisy_gray,
    0,
    255
).astype(np.uint8)



plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original RGB")

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(noisy_rgb,cv2.COLOR_BGR2RGB))
plt.title("Noisy RGB")

plt.subplot(2,2,3)
plt.imshow(gray,cmap="gray")
plt.title("Original Gray")

plt.subplot(2,2,4)
plt.imshow(noisy_gray,cmap="gray")
plt.title("Noisy Gray")

plt.show()