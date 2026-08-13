import cv2
import numpy as np

image = cv2.imread("image.jpg")

H = np.array([
    [1, 0.2, 0],
    [0.1, 1, 0],
    [0.001, 0.002, 1]
], dtype=float)

H_inv = np.linalg.inv(H)

height, width = image.shape[:2]
warped=np.zeros_like(image)

def manual_warp(image, H):

    height, width = image.shape[:2]

    warped = np.zeros_like(image)

    H_inv = np.linalg.inv(H)

    for y in range(height):

        for x in range(width):

            destination = np.array(
                [x, y, 1],
                dtype=float
            )

            source = H_inv @ destination

            source = source / source[2]

            src_x = int(source[0])
            src_y = int(source[1])

            if (
                0 <= src_x < width and
                0 <= src_y < height
            ):

                warped[y, x] = image[src_y, src_x]

    return warped

opencv_warp = cv2.warpPerspective(
    image,
    H,
    (width, height)
)

manual = manual_warp(image, H)

opencv = cv2.warpPerspective(
    image,
    H,
    (width,height)
)

cv2.imshow("Original", image)
cv2.imshow("Manual Warp", manual)
cv2.imshow("OpenCV Warp", opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()