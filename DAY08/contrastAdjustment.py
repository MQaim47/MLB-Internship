import numpy as np
import gradio as gr

def contrast_adjustment(image, alpha, beta):

    img = np.array(image)

    result = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                value = alpha * img[i, j, k] + beta

                if value > 255:
                    value = 255

                if value < 0:
                    value = 0

                result[i, j, k] = value

    return result


def process(image, alpha, beta):

    if image is None:
        return None, None, None

    original = image.copy()

    low_contrast = contrast_adjustment(
        image,
        alpha * 0.5,
        beta
    )

    high_contrast = contrast_adjustment(
        image,
        alpha * 1.5,
        beta
    )

    return original, low_contrast, high_contrast


interface = gr.Interface(
    fn=process,

    inputs=[
        gr.Image(type="numpy", label="Upload Image"),

        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Alpha (Contrast)"
        ),

        gr.Slider(
            minimum=-100,
            maximum=100,
            value=0,
            step=1,
            label="Beta (Brightness)"
        )
    ],

    outputs=[
        gr.Image(label="Original Image"),
        gr.Image(label="Low Contrast"),
        gr.Image(label="High Contrast")
    ],

    title="Contrast Adjustment"
)

interface.launch()