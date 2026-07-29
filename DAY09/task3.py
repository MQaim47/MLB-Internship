import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg",0)


def convolution(image,kernel):

    kh,kw = kernel.shape

    pad = kh//2

    padded = np.pad(
        image,
        pad,
        mode="constant"
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            region = padded[
                i:i+kh,
                j:j+kw
            ]

            output[i,j] = np.sum(
                region*kernel
            )

    return np.clip(
        output,
        0,
        255
    ).astype(np.uint8)


gaussian_kernel = np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
],dtype=np.float32)/16

gaussian_blur = convolution(
    img,
    gaussian_kernel
)


motion_kernel = np.zeros((9,9))

motion_kernel[4,:] = 1

motion_kernel /= 9

motion_blur = convolution(
    img,
    motion_kernel
)


def median_blur(image,k=3):

    pad = k//2

    padded = np.pad(
        image,
        pad,
        mode='constant'
    )

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            region = padded[
                i:i+k,
                j:j+k
            ]

            output[i,j] = np.median(region)

    return output

median_img = median_blur(img)


plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(gaussian_blur,cmap='gray')
plt.title("Gaussian Blur")

plt.subplot(2,2,3)
plt.imshow(motion_blur,cmap='gray')
plt.title("Motion Blur")

plt.subplot(2,2,4)
plt.imshow(median_img,cmap='gray')
plt.title("Median Blur")

plt.show()