import numpy as np
import gradio as gr

def brightness_adjustment(image, brightness):

    img = np.array(image)

    brighter = img.copy()
    darker = img.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                value = int(img[i, j, k]) + brightness

                if value > 255:
                    value = 255

                brighter[i, j, k] = value

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                value = int(img[i, j, k]) - brightness

                if value < 0:
                    value = 0

                darker[i, j, k] = value

    return img, brighter, darker
def process(image, brightness):

    if image is None:
        return None, None, None

    original = image.copy()

    brighter = brightness_adjustment(image, brightness)

    darker = brightness_adjustment(image, -brightness)

    return original, brighter, darker

interface = gr.Interface(
    fn=brightness_adjustment,

    inputs=[
        gr.Image(type="numpy", label="Upload Image"),

        gr.Slider(
            minimum=0,
            maximum=100,
            value=50,
            step=1,
            label="Brightness Value"
        )
    ],

    outputs=[
        gr.Image(label="Original Image"),
        gr.Image(label="Brighter Image"),
        gr.Image(label="Darker Image")
    ],

    title="Brightness Adjustment Using NumPy"
)

interface.launch()