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

    normalized = (image - min_val) / (max_val - min_val)

    normalized = normalized * 255

    return normalized.astype(np.uint8)



def sobel_edge_detection(image):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    gx = convolution(image, sobel_x)

    gy = convolution(image, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)

    return gx, gy, magnitude



image = cv2.imread("image.jpg", 0)

if image is None:
    print("Image not found")
    exit()

gx, gy, magnitude = sobel_edge_detection(image)

gx_display = normalize_image(gx)
gy_display = normalize_image(gy)
edge_display = normalize_image(magnitude)



plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gx_display, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(gy_display, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(edge_display, cmap="gray")
plt.title("Final Sobel Edge")
plt.axis("off")

plt.tight_layout()
plt.show()