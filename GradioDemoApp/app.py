from PIL import Image
import numpy as np
import gradio as gr

def manual_convolution(image):

    img = np.array(image.convert("L"))

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ]) / 9

    h,w = img.shape

    output = np.zeros((h-2,w-2))

    for i in range(h-2):

        for j in range(w-2):

            total = 0

            for ki in range(3):

                for kj in range(3):

                    total += (
                        img[i+ki][j+kj]
                        * kernel[ki][kj]
                    )

            output[i][j] = total

    output = np.clip(output,0,255)

    return Image.fromarray(
        output.astype(np.uint8)
    )

demo = gr.Interface(
    fn=manual_convolution,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Manual Convolution"
)

demo.launch()