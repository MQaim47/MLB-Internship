import cv2
import numpy as np

def get_circle_pixels(image, x, y):
    circle = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3)
    ]

    values = []

    for dx, dy in circle:
        values.append(image[y + dy, x + dx])

    return values

def check_fast_corner(circle_values, center, threshold, n=12):

    bright = []

    for value in circle_values:
        bright.append(value > center + threshold)

    dark = []

    for value in circle_values:
        dark.append(value < center - threshold)

    bright = bright + bright
    dark = dark + dark

    count = 0

    for value in bright:
        if value:
            count += 1
            if count >= n:
                return True
        else:
            count = 0

    count = 0

    for value in dark:
        if value:
            count += 1
            if count >= n:
                return True
        else:
            count = 0

    return False

def fast_detector(gray, threshold=20):

    h, w = gray.shape

    keypoints = []

    for y in range(3, h - 3):
        for x in range(3, w - 3):

            center = gray[y, x]

            circle_values = get_circle_pixels(
                gray,
                x,
                y
            )

            if check_fast_corner(
                circle_values,
                center,
                threshold
            ):
                keypoints.append((x, y))

    return keypoints

def draw_keypoints(image, keypoints):

    output = image.copy()

    for x, y in keypoints:
        cv2.circle(
            output,
            (x, y),
            2,
            (0, 0, 255),
            -1
        )

    return output

image = cv2.imread("image.jpg")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

thresholds = [10, 20, 30, 50]

for threshold in thresholds:

    keypoints = fast_detector(
        gray,
        threshold
    )

    output = draw_keypoints(
        image,
        keypoints
    )

    print(
        "Threshold:",
        threshold,
        "Keypoints:",
        len(keypoints)
    )

    cv2.imshow(
        f"FAST {threshold}",
        output
    )

cv2.waitKey(0)
cv2.destroyAllWindows()