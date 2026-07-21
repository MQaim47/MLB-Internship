from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = Image.open("image.jpg").convert("L")
img = np.array(img)


print("STEP 1: ORIGINAL IMAGE")


print("Shape:", img.shape)


def zero_padding(image, padding):

    height = image.shape[0]
    width = image.shape[1]

    padded = np.zeros(
        (
            height + (2 * padding),
            width + (2 * padding)
        ),
        dtype=np.uint8
    )

    for i in range(height):

        for j in range(width):

            padded[i + padding][j + padding] = image[i][j]

    return padded

padding_size = 1

padded_img = zero_padding(
    img,
    padding_size
)


print("STEP 2: PADDED IMAGE")


print("Padding =", padding_size)
print("Shape:", padded_img.shape)


blur_kernel = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
], dtype=float)

blur_kernel = blur_kernel / 9

print("STEP 3: BLUR KERNEL")

print(blur_kernel)



def manual_convolution(image,
                    kernel,
                    stride):

    image_height = image.shape[0]
    image_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    output_height = (
        (image_height - kernel_height)
        // stride
    ) + 1

    output_width = (
        (image_width - kernel_width)
        // stride
    ) + 1

    output = np.zeros(
        (output_height, output_width)
    )

    output_row = 0

    for i in range(
        0,
        image_height - kernel_height + 1,
        stride
    ):

        output_col = 0

        for j in range(
            0,
            image_width - kernel_width + 1,
            stride
        ):

            total = 0

            for ki in range(kernel_height):

                for kj in range(kernel_width):

                    total += (
                        image[i + ki][j + kj]
                        *
                        kernel[ki][kj]
                    )

            output[output_row][output_col] = total

            output_col += 1

        output_row += 1

    output = np.clip(
        output,
        0,
        255
    )

    return output.astype(np.uint8)

stride1_output = manual_convolution(
    padded_img,
    blur_kernel,
    1
)

print("STEP 5: STRIDE = 1")

print("Output Shape:")
print(stride1_output.shape)


stride2_output = manual_convolution(
    padded_img,
    blur_kernel,
    2
)

print("STEP 6: STRIDE = 2")

print("Output Shape:")
print(stride2_output.shape)



Image.fromarray(
    padded_img
).save(
    "padded_image.jpg"
)

Image.fromarray(
    stride1_output
).save(
    "stride1_output.jpg"
)

Image.fromarray(
    stride2_output
).save(
    "stride2_output.jpg"
)

print("STEP 7: IMAGES SAVED")



plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.imshow(
    img,
    cmap="gray"
)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(
    padded_img,
    cmap="gray"
)
plt.title("Padded Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(
    stride1_output,
    cmap="gray"
)
plt.title("Convolution + Stride 1")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(
    stride2_output,
    cmap="gray"
)
plt.title("Convolution + Stride 2")
plt.axis("off")

plt.tight_layout()
plt.show()

print("PIPELINE COMPLETED")
