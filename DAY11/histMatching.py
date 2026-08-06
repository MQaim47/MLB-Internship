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


def histogram_matching(source, reference):

    hist_source = calculate_histogram(source)

    hist_reference = calculate_histogram(reference)

    pdf_source = calculate_pdf(hist_source)
    pdf_reference = calculate_pdf(hist_reference)

    cdf_source = calculate_cdf(pdf_source)
    cdf_reference = calculate_cdf(pdf_reference)

    lookup_table = np.zeros(256, dtype=np.uint8)

    for src_intensity in range(256):

        difference = np.abs(
            cdf_source[src_intensity] -
            cdf_reference
        )

        best_match = np.argmin(difference)

        lookup_table[src_intensity] = best_match

    rows, cols = source.shape

    matched = np.zeros_like(source)

    for i in range(rows):
        for j in range(cols):
            matched[i, j] = lookup_table[
                source[i, j]
            ]

    return matched



def show_results(
        source,
        reference,
        matched):

    hist_source = calculate_histogram(source)
    hist_reference = calculate_histogram(reference)
    hist_matched = calculate_histogram(matched)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.imshow(source, cmap='gray')
    plt.title("Source Image")
    plt.axis("off")

    plt.subplot(3, 2, 2)
    plt.plot(hist_source)
    plt.title("Source Histogram")

    plt.subplot(3, 2, 3)
    plt.imshow(reference, cmap='gray')
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(3, 2, 4)
    plt.plot(hist_reference)
    plt.title("Reference Histogram")

    plt.subplot(3, 2, 5)
    plt.imshow(matched, cmap='gray')
    plt.title("Matched Image")
    plt.axis("off")

    plt.subplot(3, 2, 6)
    plt.plot(hist_matched)
    plt.title("Matched Histogram")

    plt.tight_layout()
    plt.show()



source = cv2.imread(
    "images/source.jpg",
    cv2.IMREAD_GRAYSCALE
)

reference = cv2.imread(
    "images/reference.jpg",
    cv2.IMREAD_GRAYSCALE
)

if source is None:
    print("Source image not found")
    exit()

if reference is None:
    print("Reference image not found")
    exit()

matched_image = histogram_matching(
    source,
    reference
)

show_results(
    source,
    reference,
    matched_image
)

print("Histogram Matching Completed")