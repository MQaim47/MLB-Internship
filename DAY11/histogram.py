import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(image):
    """
    Manually calculate grayscale histogram
    """
    histogram = np.zeros(256, dtype=int)

    for pixel in image.flatten():
        histogram[pixel] += 1

    return histogram


def analyze_image(histogram):
    """
    Analyze histogram and return observation
    """

    total_pixels = np.sum(histogram)

    mean_intensity = 0

    for i in range(256):
        mean_intensity += i * histogram[i]

    mean_intensity /= total_pixels

    intensity_values = np.arange(256)

    variance = np.sum(
        histogram * ((intensity_values - mean_intensity) ** 2)
    ) / total_pixels

    std_dev = np.sqrt(variance)

    observation = ""

    # Brightness Analysis
    if mean_intensity < 85:
        observation += "Dark Image\n"
    elif mean_intensity > 170:
        observation += "Bright Image\n"
    else:
        observation += "Normal Brightness\n"

    # Contrast Analysis
    if std_dev < 40:
        observation += "Low Contrast Image"
    else:
        observation += "High Contrast Image"

    return observation




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

    observation = analyze_image(histogram)

    print("\n===================================")
    print("Image:", path)
    print(observation)

    # Display Image
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(histogram)
    plt.title("Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()