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

    output = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    for i in range(rows):
        for j in range(cols):

            region = padded[
                i:i+k_h,
                j:j+k_w
            ]

            output[i, j] = np.sum(
                region * kernel
            )

    return output



def normalize_image(image):

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    image = (
        image - min_val
    ) / (
        max_val - min_val
    )

    image = image * 255

    return image.astype(np.uint8)


def gaussian_kernel(size, sigma):

    kernel = np.zeros(
        (size, size),
        dtype=np.float32
    )

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



def gaussian_blur(
        image,
        sigma):

    kernel = gaussian_kernel(
        size=5,
        sigma=sigma
    )

    return convolution(
        image,
        kernel
    )



def dog_edge_detection(image):

    blur1 = gaussian_blur(
        image,
        sigma=1
    )

    blur2 = gaussian_blur(
        image,
        sigma=2
    )

    dog = blur1 - blur2

    dog = np.abs(dog)

    return blur1, blur2, dog



image = cv2.imread(
    "image.jpg",
    0
)

if image is None:
    print("Image not found")
    exit()

blur1, blur2, dog = dog_edge_detection(image)

blur1 = normalize_image(blur1)
blur2 = normalize_image(blur2)
dog = normalize_image(dog)


plt.figure(figsize=(14,5))

plt.subplot(1,4,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(blur1, cmap="gray")
plt.title("Gaussian Sigma=1")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(blur2, cmap="gray")
plt.title("Gaussian Sigma=2")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(dog, cmap="gray")
plt.title("DoG Output")
plt.axis("off")

plt.tight_layout()
plt.show()