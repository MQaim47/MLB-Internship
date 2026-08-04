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



def gaussian_kernel(size=5, sigma=1):

    kernel = np.zeros((size, size))

    center = size // 2

    total = 0

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            value = np.exp(
                -(x**2 + y**2) /
                (2 * sigma**2)
            )

            kernel[i, j] = value

            total += value

    kernel = kernel / total

    return kernel


def gaussian_blur(image):

    kernel = gaussian_kernel(
        size=5,
        sigma=1
    )

    blurred = convolution(image, kernel)

    return blurred



def laplacian(image):

    kernel = np.array([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
    ])

    result = convolution(image, kernel)

    return np.abs(result)



def log_edge_detection(image):

    blurred = gaussian_blur(image)

    log_result = laplacian(blurred)

    return blurred, log_result


image = cv2.imread("image.jpg", 0)

if image is None:
    print("Image not found")
    exit()

blurred, log_result = log_edge_detection(image)

blurred_display = normalize_image(blurred)

log_display = normalize_image(log_result)



plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(blurred_display, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(log_display, cmap="gray")
plt.title("LoG")
plt.axis("off")

plt.tight_layout()
plt.show()