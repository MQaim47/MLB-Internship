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
# """
# Module 6 - Image Enhancement
# Single Gradio app containing all tasks (Task 1 - Task 7) as separate tabs.
# All image processing is done manually using NumPy (no cv2 filter/blur/noise
# functions), as required by the assignment.
# """

def convolve_channel(channel, kernel):
    """Manually convolve a single 2D channel with a kernel (no cv2.filter2D)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros_like(channel, dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + channel.shape[0], j:j + channel.shape[1]]

    return out


def convolve_image(img, kernel):
    """Apply convolve_channel to every channel of an RGB image and clip to 0-255."""
    img = img.astype(np.float64)
    result = np.zeros_like(img)

    for c in range(img.shape[2]):
        result[:, :, c] = convolve_channel(img[:, :, c], kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def window_stack(channel, ksize):
    """Build a stack of shifted windows for a channel (used for median filtering)."""
    pad = ksize // 2
    padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="constant")

    stack = []
    for i in range(ksize):
        for j in range(ksize):
            stack.append(padded[i:i + channel.shape[0], j:j + channel.shape[1]])

    return np.stack(stack, axis=0)



def brightness_adjustment(image, brightness):
    if image is None:
        return None, None, None

    img = np.array(image).astype(np.int16)

    brighter = np.clip(img + brightness, 0, 255).astype(np.uint8)
    darker = np.clip(img - brightness, 0, 255).astype(np.uint8)

    return image, brighter, darker




def contrast_adjustment(image, alpha, beta):
    img = np.array(image).astype(np.float64)
    result = alpha * img + beta
    return np.clip(result, 0, 255).astype(np.uint8)


def contrast_process(image, alpha, beta):
    if image is None:
        return None, None, None

    low_contrast = contrast_adjustment(image, alpha * 0.5, beta)
    high_contrast = contrast_adjustment(image, alpha * 1.5, beta)

    return image, low_contrast, high_contrast




SHARPEN_KERNEL = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


def sharpen_image(image):
    if image is None:
        return None, None

    img = np.array(image)
    sharpened = convolve_image(img, SHARPEN_KERNEL)

    return img, sharpened




def add_salt_noise(img, percent):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int((percent / 100) * h * w)

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    noisy[ys, xs] = 255

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

# demo.launch()
def add_pepper_noise(img, percent):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int((percent / 100) * h * w)

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    noisy[ys, xs] = 0

    return noisy


def add_salt_pepper_noise(img, percent):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int((percent / 100) * h * w)

    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    half = num_pixels // 2

    noisy[ys[:half], xs[:half]] = 255
    noisy[ys[half:], xs[half:]] = 0

    return noisy


def add_gaussian_noise(img, percent):
    sigma = (percent / 100) * 50  
    noise = np.random.normal(0, sigma, img.shape)

    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_noise(image, noise_percentage):
    if image is None:
        return None, None, None, None, None

    img = np.array(image)

    salt = add_salt_noise(img, noise_percentage)
    pepper = add_pepper_noise(img, noise_percentage)
    salt_pepper = add_salt_pepper_noise(img, noise_percentage)
    gaussian = add_gaussian_noise(img, noise_percentage)

    return img, salt, pepper, salt_pepper, gaussian




def mean_filter(img, ksize=3):
    kernel = np.ones((ksize, ksize)) / (ksize * ksize)
    return convolve_image(img, kernel)


def median_filter(img, ksize=3):
    result = np.zeros_like(img)

    for c in range(img.shape[2]):
        stack = window_stack(img[:, :, c], ksize)
        result[:, :, c] = np.median(stack, axis=0)

    return result.astype(np.uint8)


def gaussian_kernel(ksize=3, sigma=1.0):
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def gaussian_filter_manual(img, ksize=3, sigma=1.0):
    kernel = gaussian_kernel(ksize, sigma)
    return convolve_image(img, kernel)


def noise_reduction_process(image):
    if image is None:
        return None, None, None, None

    img = np.array(image)

    mean_result = mean_filter(img, 3)
    median_result = median_filter(img, 3)
    gaussian_result = gaussian_filter_manual(img, 3, 1.0)

    return img, mean_result, median_result, gaussian_result


def deblur_process(image):
    if image is None:
        return None, None, None

    img = np.array(image)

    blur_kernel = np.ones((5, 5)) / 25
    blurred = convolve_image(img, blur_kernel)

    restored = convolve_image(blurred, SHARPEN_KERNEL)

    return img, blurred, restored




def enhancement_comparison_process(image):
    if image is None:
        return None, None, None, None, None

    img = np.array(image)

    _, brightness_result, _ = brightness_adjustment(img, 40)
    contrast_result = contrast_adjustment(img, 1.5, 20)
    sharpen_result = convolve_image(img, SHARPEN_KERNEL)
    denoise_result = mean_filter(img, 3)

    return img, brightness_result, contrast_result, sharpen_result, denoise_result



with gr.Blocks(title="Module 6 - Image Enhancement") as demo:

    gr.Markdown("# Module 6 - Image Enhancement (All Tasks)")

   
    with gr.Tab("Task 1 - Brightness"):

        image1 = gr.Image(type="numpy", label="Upload Image")
        brightness = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Brightness Value")
        btn1 = gr.Button("Apply")

        out1 = gr.Image(label="Original")
        out2 = gr.Image(label="Brighter")
        out3 = gr.Image(label="Darker")

        btn1.click(fn=brightness_adjustment, inputs=[image1, brightness], outputs=[out1, out2, out3])

    with gr.Tab("Task 2 - Contrast"):

        image2 = gr.Image(type="numpy", label="Upload Image")
        alpha = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Alpha")
        beta = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Beta")
        btn2 = gr.Button("Apply")

        out21 = gr.Image(label="Original")
        out22 = gr.Image(label="Low Contrast")
        out23 = gr.Image(label="High Contrast")

        btn2.click(fn=contrast_process, inputs=[image2, alpha, beta], outputs=[out21, out22, out23])

    with gr.Tab("Task 3 - Sharpening"):

        image3 = gr.Image(type="numpy", label="Upload Image")
        btn3 = gr.Button("Apply")

        out31 = gr.Image(label="Original")
        out32 = gr.Image(label="Sharpened")

        btn3.click(fn=sharpen_image, inputs=image3, outputs=[out31, out32])

 
    with gr.Tab("Task 4 - Noise Generation"):

        image4 = gr.Image(type="numpy", label="Upload Image")
        noise_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Noise Percentage")
        btn4 = gr.Button("Generate Noise")

        out41 = gr.Image(label="Original")
        out42 = gr.Image(label="Salt")
        out43 = gr.Image(label="Pepper")
        out44 = gr.Image(label="Salt & Pepper")
        out45 = gr.Image(label="Gaussian")

        btn4.click(fn=generate_noise, inputs=[image4, noise_slider], outputs=[out41, out42, out43, out44, out45])

 
    with gr.Tab("Task 5 - Noise Reduction"):

        image5 = gr.Image(type="numpy", label="Upload Noisy Image")
        btn5 = gr.Button("Remove Noise")

        out51 = gr.Image(label="Original")
        out52 = gr.Image(label="Mean Filter")
        out53 = gr.Image(label="Median Filter")
        out54 = gr.Image(label="Gaussian Filter")

        btn5.click(fn=noise_reduction_process, inputs=image5, outputs=[out51, out52, out53, out54])


    with gr.Tab("Task 6 - Deblurring"):

        image6 = gr.Image(type="numpy", label="Upload Image")
        btn6 = gr.Button("Apply")

        out61 = gr.Image(label="Original")
        out62 = gr.Image(label="Blurred")
        out63 = gr.Image(label="Restored")

        btn6.click(fn=deblur_process, inputs=image6, outputs=[out61, out62, out63])

    with gr.Tab("Task 7 - Enhancement Comparison"):

        gr.Markdown(
            "Applies **brightness, contrast, sharpening, and denoising** to a "
            "low-quality image so the effect of each enhancement can be compared "
            "side by side."
        )

        image7 = gr.Image(type="numpy", label="Upload Image")
        btn7 = gr.Button("Compare")

        out71 = gr.Image(label="Original")
        out72 = gr.Image(label="Brightness")
        out73 = gr.Image(label="Contrast")
        out74 = gr.Image(label="Sharpening")
        out75 = gr.Image(label="Denoising")

        btn7.click(fn=enhancement_comparison_process, inputs=image7, outputs=[out71, out72, out73, out74, out75])


if __name__ == "__main__":
    demo.launch()

