import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(channel):
    """
    Manual histogram calculation for one channel
    """

    histogram = np.zeros(256, dtype=np.int32)

    rows, cols = channel.shape

    for i in range(rows):
        for j in range(cols):
            pixel = channel[i, j]
            histogram[pixel] += 1

    return histogram


def display_results(image_rgb, hist_r, hist_g, hist_b, image_name):

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.plot(hist_r, color='red')
    plt.title("Red Channel Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    # Green Histogram
    plt.subplot(2, 2, 3)
    plt.plot(hist_g, color='green')
    plt.title("Green Channel Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    # Blue Histogram
    plt.subplot(2, 2, 4)
    plt.plot(hist_b, color='blue')
    plt.title("Blue Channel Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    plt.suptitle(image_name)
    plt.tight_layout()
    plt.show()


image_paths = [
    "images/forest.jpg",
    "images/sky.jpg",
    "images/sunset.jpg"
]

for path in image_paths:

    image = cv2.imread(path)

    if image is None:
        print(f"Could not load {path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red_channel = image_rgb[:, :, 0]
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]

    hist_r = calculate_histogram(red_channel)
    hist_g = calculate_histogram(green_channel)
    hist_b = calculate_histogram(blue_channel)

    
    print("Image:", path)

    print("Most Frequent Red Intensity:",
          np.argmax(hist_r))

    print("Most Frequent Green Intensity:",
          np.argmax(hist_g))

    print("Most Frequent Blue Intensity:",
          np.argmax(hist_b))

    display_results(
        image_rgb,
        hist_r,
        hist_g,
        hist_b,
        path
    )