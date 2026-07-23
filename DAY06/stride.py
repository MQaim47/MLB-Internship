from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = Image.open("image.jpg").convert("L")
img = np.array(img)

print("Original Image Shape:", img.shape)


kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=float)

kernel = kernel / 9



def convolve_stride(image, kernel, stride):

    img_h = image.shape[0]
    img_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    output_h = ((img_h - kernel_h) // stride) + 1
    output_w = ((img_w - kernel_w) // stride) + 1

    output = np.zeros((output_h, output_w))

    out_row = 0

    for i in range(0, img_h - kernel_h + 1, stride):

        out_col = 0

        for j in range(0, img_w - kernel_w + 1, stride):

            total = 0

            for ki in range(kernel_h):

                for kj in range(kernel_w):

                    total += (
                        image[i + ki][j + kj]
                        * kernel[ki][kj]
                    )

            output[out_row][out_col] = total

            out_col += 1

        out_row += 1

    output = np.clip(output, 0, 255)

    return output.astype(np.uint8)


stride1_output = convolve_stride(
    img,
    kernel,
    stride=1
)



stride2_output = convolve_stride(
    img,
    kernel,
    stride=2
)



print("\nStride 1 Output Shape:")
print(stride1_output.shape)

print("\nStride 2 Output Shape:")
print(stride2_output.shape)



Image.fromarray(stride1_output).save(
    "stride1_output.jpg"
)

Image.fromarray(stride2_output).save(
    "stride2_output.jpg"
)

print("\nImages Saved Successfully")


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(stride1_output, cmap="gray")
plt.title("Stride = 1")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(stride2_output, cmap="gray")
plt.title("Stride = 2")
plt.axis("off")

plt.tight_layout()
plt.show()