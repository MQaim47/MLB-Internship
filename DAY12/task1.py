import cv2
import numpy as np
import os

def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )

    output = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):

            region = padded[
                y:y+kh,
                x:x+kw
            ]

            output[y, x] = np.sum(region * kernel)

    return output


def calculate_gradients(gray):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])

    gx = convolution(gray, sobel_x)
    gy = convolution(gray, sobel_y)

    return gx, gy


def detect_keypoints(gray, threshold_ratio=0.25):

    gx, gy = calculate_gradients(gray)

    magnitude = np.sqrt(gx**2 + gy**2)

    threshold = threshold_ratio * magnitude.max()

    keypoints = magnitude > threshold

    return keypoints, magnitude

def draw_keypoints(image, keypoints):

    output = image.copy()

    ys, xs = np.where(keypoints)

    for y, x in zip(ys, xs):
        cv2.circle(
            output,
            (x, y),
            2,
            (0, 0, 255),
            -1
        )

    return output



def process_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Cannot load {image_path}")
        return

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, magnitude = detect_keypoints(
        gray,
        threshold_ratio=0.25
    )

    output = draw_keypoints(
        image,
        keypoints
    )

    filename = os.path.basename(image_path)

    cv2.imshow(
        f"Keypoints - {filename}",
        output
    )

    print(f"\nImage: {filename}")
    print(f"Total Keypoints: {np.sum(keypoints)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



images = [
    "image.jpg",
]

for img_path in images:
    process_image(img_path)