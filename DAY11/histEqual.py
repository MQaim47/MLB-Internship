import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(image):

    histogram = np.zeros(256, dtype=np.int32)

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            histogram[image[i, j]] += 1

    return histogram



def calculate_pdf(histogram):

    total_pixels = np.sum(histogram)

    pdf = histogram / total_pixels

    return pdf



def calculate_cdf(pdf):

    cdf = np.zeros(256)

    cdf[0] = pdf[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    return cdf


def histogram_equalization(image):

    histogram = calculate_histogram(image)

    pdf = calculate_pdf(histogram)

    cdf = calculate_cdf(pdf)

    lookup_table = np.round(
        cdf * 255
    ).astype(np.uint8)

    rows, cols = image.shape

    equalized = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            equalized[i, j] = lookup_table[
                image[i, j]
            ]

    return equalized


def show_results(
        original,
        equalized,
        hist_original,
        hist_equalized):

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.plot(hist_original)
    plt.title("Original Histogram")

    plt.subplot(2, 2, 3)
    plt.imshow(equalized, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.plot(hist_equalized)
    plt.title("Equalized Histogram")

    plt.tight_layout()
    plt.show()



image = cv2.imread(
    "images/low_contrast.jpg",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("Image not found")
    exit()

equalized_image = histogram_equalization(
    image
)

hist_original = calculate_histogram(
    image
)

hist_equalized = calculate_histogram(
    equalized_image
)

show_results(
    image,
    equalized_image,
    hist_original,
    hist_equalized
)

print("Histogram Equalization Completed")