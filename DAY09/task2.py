import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg",0)

noise = np.random.normal(
    0,
    25,
    img.shape
)

img = img.astype(np.float32)

noisy = img + noise

noisy = np.clip(noisy,0,255)



def mean_filter(image,k=3):

    pad = k//2

    padded = np.pad(
        image,
        pad,
        mode="constant"
    )

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            region = padded[
                i:i+k,
                j:j+k
            ]

            output[i,j] = np.mean(region)

    return output




def gaussian_filter(image):

    kernel = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ],dtype=np.float32)

    kernel = kernel/16

    padded = np.pad(
        image,
        1,
        mode="constant"
    )

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            region = padded[
                i:i+3,
                j:j+3
            ]

            output[i,j] = np.sum(
                region * kernel
            )

    return output


mean_img = mean_filter(noisy)

gaussian_img = gaussian_filter(noisy)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(noisy,cmap='gray')
plt.title("Noisy")

plt.subplot(1,3,2)
plt.imshow(mean_img,cmap='gray')
plt.title("Mean Filter")

plt.subplot(1,3,3)
plt.imshow(gaussian_img,cmap='gray')
plt.title("Gaussian Filter")

plt.show()