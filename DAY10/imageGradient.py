import cv2
import numpy as np
import matplotlib.pyplot as plt



def convolution(image, kernel):

    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )

    rows, cols = image.shape

    output = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+k_h, j:j+k_w]

            output[i, j] = np.sum(region * kernel)

    return output



def normalize_image(image):

    min_val = np.min(image)
    max_val = np.max(image)

    normalized = (image - min_val) / (max_val - min_val)

    normalized = normalized * 255

    return normalized.astype(np.uint8)


def calculate_gradient(image):

    gx_kernel = np.array([
        [-1, 0, 1]
    ])

    gy_kernel = np.array([
        [-1],
        [ 0],
        [ 1]
    ])

    gx = convolution(image, gx_kernel)

    gy = convolution(image, gy_kernel)

    magnitude = np.sqrt(gx**2 + gy**2)

    direction = np.arctan2(gy, gx)

    return gx, gy, magnitude, direction



image = cv2.imread("image.jpg", 0)

if image is None:
    print("Image not found")
    exit()

gx, gy, magnitude, direction = calculate_gradient(image)

gx_display = normalize_image(gx)
gy_display = normalize_image(gy)
magnitude_display = normalize_image(magnitude)
direction_display = normalize_image(direction)



plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gx_display, cmap="gray")
plt.title("Gx")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gy_display, cmap="gray")
plt.title("Gy")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(magnitude_display, cmap="gray")
plt.title("Magnitude")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(direction_display, cmap="gray")
plt.title("Direction")
plt.axis("off")

plt.tight_layout()
plt.show()