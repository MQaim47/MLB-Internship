import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread(r"H:\Studies\MLB-Internship\DAY03\image.jpg")

# BGR -> RGB
rgb = img[:, :, ::-1]

# Split RGB Channels
R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

# RGB -> Grayscale
gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# RGB -> HSV
rgb_float = rgb.astype(np.float32) / 255.0

r = rgb_float[:, :, 0]
g = rgb_float[:, :, 1]
b = rgb_float[:, :, 2]

max_val = np.maximum(np.maximum(r, g), b)
min_val = np.minimum(np.minimum(r, g), b)

diff = max_val - min_val

H = np.zeros_like(max_val)

mask = diff != 0

idx = (max_val == r) & mask
H[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360

idx = (max_val == g) & mask
H[idx] = 60 * ((b[idx] - r[idx]) / diff[idx]) + 120

idx = (max_val == b) & mask
H[idx] = 60 * ((r[idx] - g[idx]) / diff[idx]) + 240

S = np.zeros_like(max_val)

non_zero = max_val != 0
S[non_zero] = diff[non_zero] / max_val[non_zero]

V = max_val

HSV = np.dstack((
    H / 2,
    S * 255,
    V * 255
)).astype(np.uint8)

# RGB -> HSL
L = (max_val + min_val) / 2

S_hsl = np.zeros_like(L)

denominator = (1 - np.abs(2 * L - 1))

valid = (diff != 0) & (denominator != 0)

S_hsl[valid] = diff[valid] / denominator[valid]

HSL = np.dstack((
    H / 2,
    S_hsl * 255,
    L * 255
)).astype(np.uint8)

# Approximate LAB
L_channel = gray.copy()

A_channel = (
    rgb[:, :, 0].astype(int)
    - rgb[:, :, 1].astype(int)
)

A_channel = np.clip(
    A_channel + 128,
    0,
    255
)

B_channel = (
    rgb[:, :, 2].astype(int)
    - rgb[:, :, 1].astype(int)
)

B_channel = np.clip(
    B_channel + 128,
    0,
    255
)

LAB = np.dstack((
    L_channel,
    A_channel,
    B_channel
)).astype(np.uint8)

# Display Results
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(rgb)
plt.title("RGB")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(HSV)
plt.title("HSV")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(HSL)
plt.title("HSL")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(LAB)
plt.title("LAB")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    plt.figure(figsize=(15,8))
    ...
    plt.show()