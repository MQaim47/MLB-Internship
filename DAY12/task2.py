import cv2
import numpy as np

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


def box_filter(image, size=3):

    kernel = np.ones((size, size), dtype=np.float32)

    kernel = kernel / (size * size)

    return convolution(image, kernel)


def calculate_gradients(gray):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    gx = convolution(gray, sobel_x)
    gy = convolution(gray, sobel_y)

    return gx, gy

def harris_detector(gray,
                    k=0.04,
                    threshold_ratio=0.01):

    Ix, Iy = calculate_gradients(gray)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    Sx2 = box_filter(Ix2, 3)
    Sy2 = box_filter(Iy2, 3)
    Sxy = box_filter(Ixy, 3)

    det = (Sx2 * Sy2) - (Sxy * Sxy)

    trace = Sx2 + Sy2

    R = det - k * (trace ** 2)

    threshold = threshold_ratio * R.max()

    corners = R > threshold

    return corners, R

def shi_tomasi_detector(gray,
                        threshold_ratio=0.01):

    Ix, Iy = calculate_gradients(gray)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    Sx2 = box_filter(Ix2, 3)
    Sy2 = box_filter(Iy2, 3)
    Sxy = box_filter(Ixy, 3)

    trace = Sx2 + Sy2

    det = (Sx2 * Sy2) - (Sxy * Sxy)

    temp = np.sqrt(
        np.maximum(
            trace * trace - 4 * det,
            0
        )
    )

    lambda1 = (trace + temp) / 2

    lambda2 = (trace - temp) / 2

    R = np.minimum(
        lambda1,
        lambda2
    )

    threshold = threshold_ratio * R.max()

    corners = R > threshold

    return corners, R



def draw_corners(image,
                 corners,
                 color):

    output = image.copy()

    ys, xs = np.where(corners)

    for y, x in zip(ys, xs):

        cv2.circle(
            output,
            (x, y),
            2,
            color,
            -1
        )

    return output


image = cv2.imread("image.jpg")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

gray = gray.astype(np.float32)

harris_corners, harris_response = harris_detector(
    gray,
    k=0.04,
    threshold_ratio=0.01
)

shi_corners, shi_response = shi_tomasi_detector(
    gray,
    threshold_ratio=0.01
)

harris_output = draw_corners(
    image,
    harris_corners,
    (0, 0, 255)
)

shi_output = draw_corners(
    image,
    shi_corners,
    (0, 255, 0)
)

print("Harris Corners:",
      np.sum(harris_corners))

print("Shi-Tomasi Corners:",
      np.sum(shi_corners))

cv2.imshow(
    "Harris Corners",
    harris_output
)

cv2.imshow(
    "Shi-Tomasi Corners",
    shi_output
)

cv2.waitKey(0)
cv2.destroyAllWindows()