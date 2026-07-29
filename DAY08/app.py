import gradio as gr
import cv2
import numpy as np
import pandas as pd



def gaussian_noise(img):

    if img is None:
        return None

    img = np.array(img)

    noise = np.random.normal(
        0,
        25,
        img.shape
    )

    noisy = img.astype(np.float32) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return noisy




def mean_filter(image,k=3):

    image = np.array(image)

    if len(image.shape)==3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

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

            output[i,j] = np.mean(region)

    return output


def gaussian_filter(image):

    image = np.array(image)

    if len(image.shape)==3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    kernel = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ],dtype=np.float32)

    kernel = kernel/16

    padded = np.pad(
        image,
        1,
        mode='constant'
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            region = padded[
                i:i+3,
                j:j+3
            ]

            output[i,j] = np.sum(
                region*kernel
            )

    return output.astype(np.uint8)


def denoise(img):

    noisy = gaussian_noise(img)

    mean_img = mean_filter(noisy)

    gauss_img = gaussian_filter(noisy)

    return noisy, mean_img, gauss_img



def convolution(image,kernel):

    kh,kw = kernel.shape

    pad = kh//2

    padded = np.pad(
        image,
        pad,
        mode='constant'
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


def blur_effects(img):

    img = np.array(img)

    if len(img.shape)==3:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

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

    median_img = median_blur(img)

    return gaussian_blur, motion_blur, median_img


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



def blur_detection(img):

    img = np.array(img)

    if len(img.shape)==3:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

    kernel = np.array([
        [0,-1,0],
        [-1,4,-1],
        [0,-1,0]
    ])

    padded = np.pad(
        img,
        1,
        mode='constant'
    )

    lap = np.zeros_like(
        img,
        dtype=np.float32
    )

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            region = padded[
                i:i+3,
                j:j+3
            ]

            lap[i,j] = np.sum(
                region*kernel
            )

    score = np.var(lap)

    if score < 100:
        status = "Blurry"
    else:
        status = "Sharp"

    return f"Score = {score:.2f}\nStatus = {status}"




def blur_metrics(img):

    img = np.array(img)

    if len(img.shape)==3:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

    kernel = np.array([
        [0,-1,0],
        [-1,4,-1],
        [0,-1,0]
    ])

    padded = np.pad(
        img,
        1,
        mode='constant'
    )

    lap = np.zeros_like(
        img,
        dtype=np.float32
    )

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            region = padded[
                i:i+3,
                j:j+3
            ]

            lap[i,j] = np.sum(
                region*kernel
            )

    score = np.var(lap)

    status = (
        "Sharp"
        if score > 100
        else "Blurry"
    )

    df = pd.DataFrame({
        "Score":[score],
        "Status":[status]
    })

    return df




with gr.Blocks() as demo:

    gr.Markdown("Noise And Blur")

    with gr.Tab("Task 1 Gaussian Noise"):

        inp = gr.Image()

        out = gr.Image()

        btn = gr.Button("Add Noise")

        btn.click(
            gaussian_noise,
            inp,
            out
        )

    with gr.Tab("Task 2 Noise Removal"):

        inp2 = gr.Image()

        out1 = gr.Image()

        out2 = gr.Image()

        out3 = gr.Image()

        btn2 = gr.Button("Denoise")

        btn2.click(
            denoise,
            inp2,
            [out1,out2,out3]
        )

    with gr.Tab("Task 3 Blur Generation"):

        inp3 = gr.Image()

        g = gr.Image()

        m = gr.Image()

        med = gr.Image()

        btn3 = gr.Button("Generate Blur")

        btn3.click(
            blur_effects,
            inp3,
            [g,m,med]
        )

    with gr.Tab("Task 4 Blur Detection"):

        inp4 = gr.Image()

        result = gr.Textbox()

        btn4 = gr.Button(
            "Detect Blur"
        )

        btn4.click(
            blur_detection,
            inp4,
            result
        )

    with gr.Tab("Task 5 Blur Metrics"):

        inp5 = gr.Image()

        table = gr.Dataframe()

        btn5 = gr.Button(
            "Analyze"
        )

        btn5.click(
            blur_metrics,
            inp5,
            table
        )

demo.launch()