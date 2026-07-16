import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread(r"H:\Studies\MLB-Internship\DAY03\image.jpg")

rgb = img[:, :, ::-1]

R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

# Manual Grayscale
gray = (
    0.299 * R +
    0.587 * G +
    0.114 * B
).astype(np.uint8)


L_channel = gray.copy()

A_channel = (
    rgb[:, :, 0].astype(int)
    -
    rgb[:, :, 1].astype(int)
)

A_channel = np.clip(
    A_channel + 128,
    0,
    255
)

B_channel = (
    rgb[:, :, 2].astype(int)
    -
    rgb[:, :, 1].astype(int)
)

B_channel = np.clip(
    B_channel + 128,
    0,
    255
)


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(L_channel, cmap="gray")
plt.title("L Channel")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(A_channel, cmap="gray")
plt.title("A Channel")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(B_channel, cmap="gray")
plt.title("B Channel")
plt.axis("off")

plt.tight_layout()
plt.show()