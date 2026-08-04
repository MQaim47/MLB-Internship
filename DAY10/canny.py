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



def gaussian_kernel(size=5, sigma=1):

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
                -(x**2 + y**2)
                /
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

    return convolution(
        image,
        kernel
    )



def sobel_gradient(image):

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

    gx = convolution(
        image,
        sobel_x
    )

    gy = convolution(
        image,
        sobel_y
    )

    magnitude = np.sqrt(
        gx**2 + gy**2
    )

    direction = np.arctan2(
        gy,
        gx
    )

    return gx, gy, magnitude, direction


def non_maximum_suppression(
        magnitude,
        direction):

    rows, cols = magnitude.shape

    output = np.zeros(
        (rows, cols),
        dtype=np.float32
    )

    angle = direction * 180 / np.pi

    angle[angle < 0] += 180

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            q = 255
            r = 255

            # 0 degree
            if (0 <= angle[i,j] < 22.5) or \
               (157.5 <= angle[i,j] <= 180):

                q = magnitude[i, j+1]
                r = magnitude[i, j-1]

            # 45 degree
            elif 22.5 <= angle[i,j] < 67.5:

                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]

            # 90 degree
            elif 67.5 <= angle[i,j] < 112.5:

                q = magnitude[i+1, j]
                r = magnitude[i-1, j]

            # 135 degree
            elif 112.5 <= angle[i,j] < 157.5:

                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (
                magnitude[i,j] >= q
                and
                magnitude[i,j] >= r
            ):
                output[i,j] = magnitude[i,j]
            else:
                output[i,j] = 0

    return output


def double_threshold(
        image,
        low_ratio=0.05,
        high_ratio=0.15):

    high = image.max() * high_ratio

    low = high * low_ratio

    rows, cols = image.shape

    result = np.zeros(
        (rows, cols),
        dtype=np.uint8
    )

    weak = 75
    strong = 255

    strong_i, strong_j = np.where(
        image >= high
    )

    weak_i, weak_j = np.where(
        (image >= low)
        &
        (image < high)
    )

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong



def hysteresis(
        image,
        weak,
        strong):

    rows, cols = image.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            if image[i,j] == weak:

                if (
                    strong in image[
                        i-1:i+2,
                        j-1:j+2
                    ]
                ):
                    image[i,j] = strong
                else:
                    image[i,j] = 0

    return image



def manual_canny(image):

    blurred = gaussian_blur(image)

    gx, gy, mag, direction = \
        sobel_gradient(blurred)

    nms = non_maximum_suppression(
        mag,
        direction
    )

    thresholded, weak, strong = \
        double_threshold(nms)

    final_edges = hysteresis(
        thresholded,
        weak,
        strong
    )

    return (
        blurred,
        mag,
        nms,
        thresholded,
        final_edges
    )



image = cv2.imread(
    "image.jpg",
    0
)

if image is None:
    print("Image not found")
    exit()

(
    blurred,
    magnitude,
    nms,
    thresholded,
    canny
) = manual_canny(image)

blurred = normalize_image(blurred)
magnitude = normalize_image(magnitude)
nms = normalize_image(nms)

plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(magnitude, cmap="gray")
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(nms, cmap="gray")
plt.title("NMS")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(thresholded, cmap="gray")
plt.title("Threshold")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(canny, cmap="gray")
plt.title("Final Canny")
plt.axis("off")

plt.tight_layout()
plt.show()