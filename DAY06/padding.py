from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



img = Image.open("image.jpg").convert("L")
img = np.array(img)

print("Original Image Shape:")
print(img.shape)


def zero_padding(image, padding):

    height = image.shape[0]
    width = image.shape[1]

    padded_image = np.zeros(
        (height + 2 * padding,
         width + 2 * padding),
        dtype=np.uint8
    )

    for i in range(height):

        for j in range(width):

            padded_image[i + padding][j + padding] = image[i][j]

    return padded_image



padding_size = 1

zero_padded = zero_padding(
    img,
    padding_size
)

print("\nZero Padded Shape:")
print(zero_padded.shape)



kernel_size = 3

same_padding_size = (kernel_size - 1) // 2

same_padded = zero_padding(
    img,
    same_padding_size
)

print("\nSame Padding Size:")
print(same_padding_size)

print("\nSame Padded Shape:")
print(same_padded.shape)



Image.fromarray(zero_padded).save(
    "zero_padded.jpg"
)

Image.fromarray(same_padded).save(
    "same_padded.jpg"
)

print("\nImages Saved Successfully")



plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(zero_padded, cmap="gray")
plt.title("Zero Padding")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(same_padded, cmap="gray")
plt.title("Same Padding")
plt.axis("off")

plt.tight_layout()
plt.show()