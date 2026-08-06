import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(image):
    """
    Manual grayscale histogram calculation
    """

    histogram = np.zeros(256, dtype=np.int32)

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            pixel = image[i, j]
            histogram[pixel] += 1

    return histogram


def display_results(image, histogram, image_name):

    plt.figure(figsize=(12, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"{image_name}")
    plt.axis("off")

    # Histogram
    plt.subplot(1, 2, 2)
    plt.plot(histogram)
    plt.title("Grayscale Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])

    plt.tight_layout()
    plt.show()





image_paths = [
    "images/dark.jpg",
    "images/bright.jpg",
    "images/low_contrast.jpg",
    "images/high_contrast.jpg"
]

for path in image_paths:

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Could not load {path}")
        continue

    histogram = calculate_histogram(image)

    
    print("Image:", path)

    print("Total Pixels:", np.sum(histogram))

    print("Black Pixels (0):", histogram[0])

    print("White Pixels (255):", histogram[255])

    print("Most Frequent Intensity:",
          np.argmax(histogram))

    display_results(image, histogram, path)