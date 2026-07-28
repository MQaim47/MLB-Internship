import numpy as np
import gradio as gr


# Brightness
def brightness_adjustment(img, value):

    result = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                pixel = int(img[i,j,k]) + value

                if pixel > 255:
                    pixel = 255

                if pixel < 0:
                    pixel = 0

                result[i,j,k] = pixel

    return result


# Contrast
def contrast_adjustment(img, alpha, beta):

    result = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                pixel = alpha * img[i,j,k] + beta

                if pixel > 255:
                    pixel = 255

                if pixel < 0:
                    pixel = 0

                result[i,j,k] = pixel

    return result


# Sharpening

def sharpen_image(img):

    output = np.zeros_like(img)

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    for ch in range(img.shape[2]):

        padded = np.pad(
            img[:,:,ch],
            ((1,1),(1,1)),
            mode="constant"
        )

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                region = padded[i:i+3, j:j+3]

                value = np.sum(region * kernel)

                if value > 255:
                    value = 255

                if value < 0:
                    value = 0

                output[i,j,ch] = value

    return output


# Mean Filter Denoising
def denoise_image(img):

    output = np.zeros_like(img)

    kernel = np.ones((3,3))/9

    for ch in range(img.shape[2]):

        padded = np.pad(
            img[:,:,ch],
            ((1,1),(1,1)),
            mode="constant"
        )

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                region = padded[i:i+3, j:j+3]

                value = np.sum(region * kernel)

                output[i,j,ch] = value

    return output


# Main Function
def process(image):

    if image is None:
        return None, None, None, None, None

    img = np.array(image)

    bright = brightness_adjustment(img, 50)

    contrast = contrast_adjustment(img, 1.5, 20)

    sharpen = sharpen_image(img)

    denoise = denoise_image(img)

    return (
        img,
        bright,
        contrast,
        sharpen,
        denoise
    )



# Interface

interface = gr.Interface(
    fn=process,

    inputs=gr.Image(type="numpy"),

    outputs=[
        gr.Image(label="Original"),
        gr.Image(label="Brightness"),
        gr.Image(label="Contrast"),
        gr.Image(label="Sharpening"),
        gr.Image(label="Denoising")
    ],

    title="Image Enhancement Comparison"
)

interface.launch()