# import gradio as gr
# import numpy as np

# from gradient import calculate_gradient, normalize_image
# from sobel import sobel_edge_detection
# from laplacian import laplacian_edge_detection
# from LoG import log_edge_detection
# from DoG import dog_edge_detection
# from canny import manual_canny

# def gradient_tab(image):

#     image = np.array(image)

#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)

#     gx, gy, mag, direction = calculate_gradient(image)

#     return (
#         normalize_image(gx),
#         normalize_image(gy),
#         normalize_image(mag),
#         normalize_image(direction)
#     )



# def sobel_tab(image):

#     image = np.array(image)

#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)

#     gx, gy, mag = sobel_edge_detection(image)

#     return (
#         normalize_image(gx),
#         normalize_image(gy),
#         normalize_image(mag)
#     )

# def laplacian_tab(image):

#     image = np.array(image)

#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)

#     result = laplacian_edge_detection(image)

#     return normalize_image(result)



# def log_tab(image):

#     image = np.array(image)

#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)

#     blurred, log_result = log_edge_detection(image)

#     return (
#         normalize_image(blurred),
#         normalize_image(log_result)
#     )


# def dog_tab(image):

#     image = np.array(image)

#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)

#     blur1, blur2, dog = dog_edge_detection(image)

#     return (
#         normalize_image(blur1),
#         normalize_image(blur2),
#         normalize_image(dog)
#     )


# def canny_tab(image):

#     image = np.array(image)

#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)

#     blurred, mag, nms, threshold, canny = manual_canny(image)

#     return (
#         normalize_image(blurred),
#         normalize_image(mag),
#         normalize_image(nms),
#         threshold,
#         canny
#     )


# def comparison_tab(image):

#     image = np.array(image)

#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)

#     _, _, sobel = sobel_edge_detection(image)

#     laplacian = laplacian_edge_detection(image)

#     _, log_result = log_edge_detection(image)

#     _, _, dog = dog_edge_detection(image)

#     _, _, _, _, canny = manual_canny(image)

#     return (
#         normalize_image(sobel),
#         normalize_image(laplacian),
#         normalize_image(log_result),
#         normalize_image(dog),
#         canny
#     )



# with gr.Blocks(title="Module 9 - Edge Detection") as demo:

#     gr.Markdown("# Module 9 - Edge Detection")


#     with gr.Tab("Task 1 - Gradient"):

#         inp = gr.Image(type="numpy")

#         btn = gr.Button("Run")

#         gx = gr.Image(label="Gx")
#         gy = gr.Image(label="Gy")
#         mag = gr.Image(label="Magnitude")
#         direction = gr.Image(label="Direction")

#         btn.click(
#             gradient_tab,
#             inp,
#             [gx, gy, mag, direction]
#         )


#     with gr.Tab("Task 2 - Sobel"):

#         inp2 = gr.Image(type="numpy")

#         btn2 = gr.Button("Run")

#         sx = gr.Image(label="Sobel X")
#         sy = gr.Image(label="Sobel Y")
#         smag = gr.Image(label="Final Edge")

#         btn2.click(
#             sobel_tab,
#             inp2,
#             [sx, sy, smag]
#         )

    
#     with gr.Tab("Task 3 - Laplacian"):

#         inp3 = gr.Image(type="numpy")

#         btn3 = gr.Button("Run")

#         lap = gr.Image(label="Laplacian")

#         btn3.click(
#             laplacian_tab,
#             inp3,
#             lap
#         )

#     with gr.Tab("Task 4 - LoG"):

#         inp4 = gr.Image(type="numpy")

#         btn4 = gr.Button("Run")

#         blur = gr.Image(label="Gaussian Blur")
#         log = gr.Image(label="LoG")

#         btn4.click(
#             log_tab,
#             inp4,
#             [blur, log]
#         )

#     with gr.Tab("Task 5 - DoG"):

#         inp5 = gr.Image(type="numpy")

#         btn5 = gr.Button("Run")

#         b1 = gr.Image(label="Sigma 1")
#         b2 = gr.Image(label="Sigma 2")
#         dog = gr.Image(label="DoG")

#         btn5.click(
#             dog_tab,
#             inp5,
#             [b1, b2, dog]
#         )

#     with gr.Tab("Task 6 - Canny"):

#         inp6 = gr.Image(type="numpy")

#         btn6 = gr.Button("Run")

#         cb = gr.Image(label="Blurred")
#         cm = gr.Image(label="Magnitude")
#         cn = gr.Image(label="NMS")
#         ct = gr.Image(label="Threshold")
#         cc = gr.Image(label="Final Canny")

#         btn6.click(
#             canny_tab,
#             inp6,
#             [cb, cm, cn, ct, cc]
#         )

   
#     with gr.Tab("Task 7 - Comparison"):

#         inp7 = gr.Image(type="numpy")

#         btn7 = gr.Button("Compare")

#         sobel_img = gr.Image(label="Sobel")
#         lap_img = gr.Image(label="Laplacian")
#         log_img = gr.Image(label="LoG")
#         dog_img = gr.Image(label="DoG")
#         canny_img = gr.Image(label="Canny")

#         btn7.click(
#             comparison_tab,
#             inp7,
#             [
#                 sobel_img,
#                 lap_img,
#                 log_img,
#                 dog_img,
#                 canny_img
#             ]
#         )

# demo.launch()
import gradio as gr
import numpy as np

from imageGradient import calculate_gradient, normalize_image
from sobel import sobel_edge_detection
from laplacian import laplacian_edge_detection
from LoG import log_edge_detection
from DoG import dog_edge_detection
from canny import manual_canny

def gradient_tab(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    gx, gy, mag, direction = calculate_gradient(image)

    return (
        normalize_image(gx),
        normalize_image(gy),
        normalize_image(mag),
        normalize_image(direction)
    )



def sobel_tab(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    gx, gy, mag = sobel_edge_detection(image)

    return (
        normalize_image(gx),
        normalize_image(gy),
        normalize_image(mag)
    )

def laplacian_tab(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    result = laplacian_edge_detection(image)

    return normalize_image(result)



def log_tab(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    blurred, log_result = log_edge_detection(image)

    return (
        normalize_image(blurred),
        normalize_image(log_result)
    )


def dog_tab(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    blur1, blur2, dog = dog_edge_detection(image)

    return (
        normalize_image(blur1),
        normalize_image(blur2),
        normalize_image(dog)
    )


def canny_tab(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    blurred, mag, nms, threshold, canny = manual_canny(image)

    return (
        normalize_image(blurred),
        normalize_image(mag),
        normalize_image(nms),
        threshold,
        canny
    )


def comparison_tab(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    _, _, sobel = sobel_edge_detection(image)

    laplacian = laplacian_edge_detection(image)

    _, log_result = log_edge_detection(image)

    _, _, dog = dog_edge_detection(image)

    _, _, _, _, canny = manual_canny(image)

    return (
        normalize_image(sobel),
        normalize_image(laplacian),
        normalize_image(log_result),
        normalize_image(dog),
        canny
    )



with gr.Blocks(title="Day 10 - Edge Detection") as demo:

    gr.Markdown("# Day 10 - Edge Detection")


    with gr.Tab("Task 1 - Gradient"):

        inp = gr.Image(type="numpy")

        btn = gr.Button("Run")

        gx = gr.Image(label="Gx")
        gy = gr.Image(label="Gy")
        mag = gr.Image(label="Magnitude")
        direction = gr.Image(label="Direction")

        btn.click(
            gradient_tab,
            inp,
            [gx, gy, mag, direction]
        )


    with gr.Tab("Task 2 - Sobel"):

        inp2 = gr.Image(type="numpy")

        btn2 = gr.Button("Run")

        sx = gr.Image(label="Sobel X")
        sy = gr.Image(label="Sobel Y")
        smag = gr.Image(label="Final Edge")

        btn2.click(
            sobel_tab,
            inp2,
            [sx, sy, smag]
        )

    
    with gr.Tab("Task 3 - Laplacian"):

        inp3 = gr.Image(type="numpy")

        btn3 = gr.Button("Run")

        lap = gr.Image(label="Laplacian")

        btn3.click(
            laplacian_tab,
            inp3,
            lap
        )

    with gr.Tab("Task 4 - LoG"):

        inp4 = gr.Image(type="numpy")

        btn4 = gr.Button("Run")

        blur = gr.Image(label="Gaussian Blur")
        log = gr.Image(label="LoG")

        btn4.click(
            log_tab,
            inp4,
            [blur, log]
        )

    with gr.Tab("Task 5 - DoG"):

        inp5 = gr.Image(type="numpy")

        btn5 = gr.Button("Run")

        b1 = gr.Image(label="Sigma 1")
        b2 = gr.Image(label="Sigma 2")
        dog = gr.Image(label="DoG")

        btn5.click(
            dog_tab,
            inp5,
            [b1, b2, dog]
        )

    with gr.Tab("Task 6 - Canny"):

        inp6 = gr.Image(type="numpy")

        btn6 = gr.Button("Run")

        cb = gr.Image(label="Blurred")
        cm = gr.Image(label="Magnitude")
        cn = gr.Image(label="NMS")
        ct = gr.Image(label="Threshold")
        cc = gr.Image(label="Final Canny")

        btn6.click(
            canny_tab,
            inp6,
            [cb, cm, cn, ct, cc]
        )

   
    with gr.Tab("Task 7 - Comparison"):

        inp7 = gr.Image(type="numpy")

        btn7 = gr.Button("Compare")

        sobel_img = gr.Image(label="Sobel")
        lap_img = gr.Image(label="Laplacian")
        log_img = gr.Image(label="LoG")
        dog_img = gr.Image(label="DoG")
        canny_img = gr.Image(label="Canny")

        btn7.click(
            comparison_tab,
            inp7,
            [
                sobel_img,
                lap_img,
                log_img,
                dog_img,
                canny_img
            ]
        )

demo.launch()