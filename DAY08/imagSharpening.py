import numpy as np
import gradio as gr

def sharpen_image(image):

    if image is None:
        return None, None

    img = np.array(image)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = np.zeros_like(img)

    for channel in range(img.shape[2]):

    
        padded = np.pad(
            img[:, :, channel],
            ((1, 1), (1, 1)),
            mode='constant'
        )

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                region = padded[i:i+3, j:j+3]

                value = np.sum(region * kernel)

                if value > 255:
                    value = 255

                if value < 0:
                    value = 0

                sharpened[i, j, channel] = value

    return img, sharpened


interface = gr.Interface(
    fn=sharpen_image,

    inputs=gr.Image(type="numpy"),

    outputs=[
        gr.Image(label="Original Image"),
        gr.Image(label="Sharpened Image")
    ],

    title="Manual Image Sharpening"
)

interface.launch()