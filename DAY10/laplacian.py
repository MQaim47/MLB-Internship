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

    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)

    image = image * 255

    return image.astype(np.uint8)


def laplacian_edge_detection(image):

    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ])

    laplacian = convolution(image, laplacian_kernel)

    laplacian = np.abs(laplacian)

    return laplacian



image = cv2.imread("image.jpg", 0)

if image is None:
    print("Image not found")
    exit()

laplacian = laplacian_edge_detection(image)

laplacian_display = normalize_image(laplacian)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(laplacian_display, cmap="gray")
plt.title("Laplacian Edge")
plt.axis("off")

plt.tight_layout()
plt.show()