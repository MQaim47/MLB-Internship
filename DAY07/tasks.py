# import cv2
# import numpy as np
# import gradio as gr


# # ==========================
# # PADDING
# # ==========================

# def pad_image(image, pad):

#     h, w = image.shape

#     padded = np.zeros(
#         (h + 2 * pad, w + 2 * pad),
#         dtype=np.uint8
#     )

#     padded[pad:pad+h, pad:pad+w] = image

#     return padded


# # ==========================
# # CONVOLUTION
# # ==========================

# def convolution(image, kernel):

#     k = kernel.shape[0]

#     pad = k // 2

#     padded = pad_image(image, pad)

#     h, w = image.shape

#     output = np.zeros(
#         (h, w),
#         dtype=np.uint8
#     )

#     for i in range(h):

#         for j in range(w):

#             region = padded[
#                 i:i+k,
#                 j:j+k
#             ]

#             value = np.sum(region * kernel)

#             value = np.clip(
#                 value,
#                 0,
#                 255
#             )

#             output[i, j] = value

#     return output


# # ==========================
# # MEAN FILTER
# # ==========================

# def mean_filter(image, size):

#     kernel = np.ones(
#         (size, size),
#         dtype=np.float32
#     )

#     kernel = kernel / (size * size)

#     return convolution(
#         image,
#         kernel
#     )


# # ==========================
# # BOX FILTER
# # ==========================

# def box_filter(image, size):

#     kernel = np.ones(
#         (size, size),
#         dtype=np.float32
#     )

#     kernel = kernel / np.sum(kernel)

#     return convolution(
#         image,
#         kernel
#     )


# # ==========================
# # GAUSSIAN KERNELS
# # ==========================

# def gaussian_kernel_3():

#     kernel = np.array([
#         [1, 2, 1],
#         [2, 4, 2],
#         [1, 2, 1]
#     ], dtype=np.float32)

#     return kernel / 16


# def gaussian_kernel_5():

#     kernel = np.array([
#         [1, 4, 6, 4, 1],
#         [4,16,24,16,4],
#         [6,24,36,24,6],
#         [4,16,24,16,4],
#         [1,4,6,4,1]
#     ], dtype=np.float32)

#     return kernel / np.sum(kernel)


# def gaussian_kernel_7():

#     kernel = np.array([
#         [1,6,15,20,15,6,1],
#         [6,36,90,120,90,36,6],
#         [15,90,225,300,225,90,15],
#         [20,120,300,400,300,120,20],
#         [15,90,225,300,225,90,15],
#         [6,36,90,120,90,36,6],
#         [1,6,15,20,15,6,1]
#     ], dtype=np.float32)

#     return kernel / np.sum(kernel)


# # ==========================
# # GAUSSIAN FILTER
# # ==========================

# def gaussian_filter(image, size):

#     if size == 3:

#         kernel = gaussian_kernel_3()

#     elif size == 5:

#         kernel = gaussian_kernel_5()

#     else:

#         kernel = gaussian_kernel_7()

#     return convolution(
#         image,
#         kernel
#     )


# # ==========================
# # SALT PEPPER NOISE
# # ==========================

# def add_salt_pepper(
#     image,
#     salt_prob=0.02,
#     pepper_prob=0.02
# ):

#     noisy = image.copy()

#     h, w = image.shape

#     total = h * w

#     salt_pixels = int(
#         total * salt_prob
#     )

#     pepper_pixels = int(
#         total * pepper_prob
#     )

#     for _ in range(salt_pixels):

#         x = np.random.randint(0, h)
#         y = np.random.randint(0, w)

#         noisy[x, y] = 255

#     for _ in range(pepper_pixels):

#         x = np.random.randint(0, h)
#         y = np.random.randint(0, w)

#         noisy[x, y] = 0

#     return noisy


# # ==========================
# # MEDIAN FILTER
# # ==========================

# def median_filter(image, size):

#     pad = size // 2

#     padded = pad_image(
#         image,
#         pad
#     )

#     h, w = image.shape

#     output = np.zeros(
#         (h, w),
#         dtype=np.uint8
#     )

#     for i in range(h):

#         for j in range(w):

#             region = padded[
#                 i:i+size,
#                 j:j+size
#             ]

#             pixels = []

#             for r in range(size):

#                 for c in range(size):

#                     pixels.append(
#                         int(region[r, c])
#                     )

#             pixels.sort()

#             median = pixels[
#                 len(pixels)//2
#             ]

#             output[i, j] = median

#     return output


# # ==========================
# # PROCESS FUNCTION
# # ==========================

# def process_image(
#     image,
#     filter_name,
#     kernel_size
# ):

#     if image is None:
#         return None

#     if len(image.shape) == 3:

#         gray = cv2.cvtColor(
#             image,
#             cv2.COLOR_RGB2GRAY
#         )

#     else:

#         gray = image

#     if filter_name == "Mean":

#         result = mean_filter(
#             gray,
#             kernel_size
#         )

#     elif filter_name == "Box":

#         result = box_filter(
#             gray,
#             kernel_size
#         )

#     elif filter_name == "Gaussian":

#         result = gaussian_filter(
#             gray,
#             kernel_size
#         )

#     elif filter_name == "Median":

#         noisy = add_salt_pepper(gray)

#         result = median_filter(
#             noisy,
#             kernel_size
#         )

#     else:

#         result = gray

#     return result


# # ==========================
# # EXAMPLES
# # ==========================

# examples = [
#     ["examples/sample1.jpg", "Mean", 3],
#     ["examples/sample2.jpg", "Gaussian", 5],
#     ["examples/sample3.jpg", "Median", 3]
# ]


# # ==========================
# # GRADIO INTERFACE
# # ==========================

# demo = gr.Interface(

#     fn=process_image,

#     inputs=[

#         gr.Image(
#             type="numpy",
#             label="Upload Image"
#         ),

#         gr.Dropdown(
#             choices=[
#                 "Mean",
#                 "Box",
#                 "Gaussian",
#                 "Median"
#             ],
#             value="Mean",
#             label="Filter Type"
#         ),

#         gr.Radio(
#             choices=[3, 5, 7],
#             value=3,
#             label="Kernel Size"
#         )
#     ],

#     outputs=gr.Image(
#         type="numpy",
#         label="Filtered Output"
#     ),

#     title="Manual Image Filtering",

#     description="""
#     Mean Filter
#     Box Filter
#     Gaussian Filter
#     Median Filter

#     Implemented manually using NumPy.
#     No OpenCV filtering functions used.
#     """,

#     examples=examples
# )

# demo.launch()

# import cv2
# import numpy as np
# import time


# img = cv2.imread("image.jpg", 0)




# def pad_image(image, pad):

#     h, w = image.shape

#     padded = np.zeros((h + 2 * pad, w + 2 * pad),
#                       dtype=np.uint8)

#     padded[pad:pad+h, pad:pad+w] = image

#     return padded



# def convolution(image, kernel):

#     k = kernel.shape[0]

#     pad = k // 2

#     padded = pad_image(image, pad)

#     h, w = image.shape

#     output = np.zeros((h, w),
#                       dtype=np.uint8)

#     for i in range(h):

#         for j in range(w):

#             region = padded[i:i+k, j:j+k]

#             value = np.sum(region * kernel)

#             value = max(0, min(255, value))

#             output[i, j] = value

#     return output


# # TASK 1
# # MEAN FILTER

# def mean_filter(image, size=3):

#     kernel = np.ones((size, size),
#                      dtype=np.float32)

#     kernel = kernel / (size * size)

#     return convolution(image, kernel)


# # TASK 2
# # BOX FILTER


# def box_filter(image, size=3):

#     kernel = np.ones((size, size),
#                      dtype=np.float32)

#     kernel_sum = np.sum(kernel)

#     kernel = kernel / kernel_sum

#     return convolution(image, kernel)


# # TASK 3
# # GAUSSIAN KERNEL

# def gaussian_kernel_3x3():

#     kernel = np.array([
#         [1,2,1],
#         [2,4,2],
#         [1,2,1]
#     ], dtype=np.float32)

#     kernel = kernel / 16

#     return kernel


# def gaussian_kernel_5x5():

#     kernel = np.array([
#         [1,4,6,4,1],
#         [4,16,24,16,4],
#         [6,24,36,24,6],
#         [4,16,24,16,4],
#         [1,4,6,4,1]
#     ], dtype=np.float32)

#     kernel = kernel / np.sum(kernel)

#     return kernel


# def gaussian_filter(image, size=3):

#     if size == 3:
#         kernel = gaussian_kernel_3x3()

#     elif size == 5:
#         kernel = gaussian_kernel_5x5()

#     else:
#         kernel = gaussian_kernel_3x3()

#     return convolution(image, kernel)


# # SALT PEPPER NOISE

# def add_salt_pepper(image,
#                     salt_prob=0.02,
#                     pepper_prob=0.02):

#     noisy = image.copy()

#     h, w = image.shape

#     total = h * w

#     salt_pixels = int(total * salt_prob)
#     pepper_pixels = int(total * pepper_prob)

#     # Salt

#     for _ in range(salt_pixels):

#         x = np.random.randint(0, h)
#         y = np.random.randint(0, w)

#         noisy[x, y] = 255

#     # Pepper

#     for _ in range(pepper_pixels):

#         x = np.random.randint(0, h)
#         y = np.random.randint(0, w)

#         noisy[x, y] = 0

#     return noisy


# # TASK 4
# # MEDIAN FILTER


# def median_filter(image, size=3):

#     pad = size // 2

#     padded = pad_image(image, pad)

#     h, w = image.shape

#     output = np.zeros((h, w),
#                       dtype=np.uint8)

#     for i in range(h):

#         for j in range(w):

#             region = padded[i:i+size,
#                             j:j+size]

#             pixels = []

#             for r in range(size):
#                 for c in range(size):
#                     pixels.append(region[r, c])

#             pixels.sort()

#             median = pixels[len(pixels)//2]

#             output[i, j] = median

#     return output



# # TASK 5
# # FILTER COMPARISON


# noisy = add_salt_pepper(img)

# mean_img = mean_filter(noisy)

# box_img = box_filter(noisy)

# gaussian_img = gaussian_filter(noisy)

# median_img = median_filter(noisy)

# print("\nFILTER COMPARISON\n")

# print("Mean:Good smoothing, blur high")
# print("Box:Similar to Mean")
# print("Gaussian:Better edge preservation")
# print("Median:Best for salt-pepper noise")



# # TASK 6
# # KERNEL SIZE ANALYSIS


# print("\nKERNEL SIZE ANALYSIS\n")

# for k in [3,5,7]:

#     mean_result = mean_filter(noisy, k)

#     print(f"Mean Filter {k}x{k} Done")




# print("\nTIMING ANALYSIS\n")

# start = time.time()
# mean_filter(noisy)
# mean_time = time.time() - start

# start = time.time()
# box_filter(noisy)
# box_time = time.time() - start

# start = time.time()
# gaussian_filter(noisy)
# gaussian_time = time.time() - start

# start = time.time()
# median_filter(noisy)
# median_time = time.time() - start

# print("Mean Filter     :", mean_time)
# print("Box Filter      :", box_time)
# print("Gaussian Filter :", gaussian_time)
# print("Median Filter   :", median_time)




# cv2.imshow("Original", img)

# cv2.imshow("Noisy", noisy)

# cv2.imshow("Mean", mean_img)

# cv2.imshow("Box", box_img)

# cv2.imshow("Gaussian", gaussian_img)

# cv2.imshow("Median", median_img)

# cv2.waitKey(0)

# examples = [
#     ["examples/image.jpg", "Mean", 3],
#     ["examples/image.jpg", "Gaussian", 5],
#     ["examples/image.jpg", "Median", 3]
# ]
# # Gradio userInterface  
# demo = gr.Interface(
#     fn=process_image,

#     inputs=[
#         gr.Image(label="Upload Image"),

#         gr.Dropdown(
#             choices=[
#                 "Mean",
#                 "Box",
#                 "Gaussian",
#                 "Median"
#             ],
#             value="Mean",
#             label="Filter"
#         ),

#         gr.Radio(
#             choices=[3,5,7],
#             value=3,
#             label="Kernel Size"
#         )
#     ],

#     outputs=gr.Image(
#         label="Filtered Output"
#     ),

#     title="Manual Image Filtering",

#     description="""
#     Mean Filter,
#     Box Filter,
#     Gaussian Filter,
#     Median Filter

#     Implemented Manually Using NumPy.
#     No OpenCV Filtering Functions Used.
#     """,

#     examples=examples
# )

# demo.launch()
import gradio as gr
import cv2
import numpy as np


# ==========================
# PADDING
# ==========================

def pad_image(image, pad):

    h, w = image.shape

    padded = np.zeros(
        (h + 2 * pad, w + 2 * pad),
        dtype=np.uint8
    )

    padded[pad:pad+h, pad:pad+w] = image

    return padded


# ==========================
# CONVOLUTION
# ==========================

def convolution(image, kernel):

    k = kernel.shape[0]

    pad = k // 2

    padded = pad_image(image, pad)

    h, w = image.shape

    output = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    for i in range(h):

        for j in range(w):

            region = padded[i:i+k, j:j+k]

            value = np.sum(region * kernel)

            value = max(0, min(255, value))

            output[i, j] = value

    return output


# ==========================
# MEAN FILTER
# ==========================

def mean_filter(image, size):

    kernel = np.ones(
        (size, size),
        dtype=np.float32
    )

    kernel = kernel / (size * size)

    return convolution(image, kernel)


# ==========================
# BOX FILTER
# ==========================

def box_filter(image, size):

    kernel = np.ones(
        (size, size),
        dtype=np.float32
    )

    kernel_sum = np.sum(kernel)

    kernel = kernel / kernel_sum

    return convolution(image, kernel)


# ==========================
# GAUSSIAN KERNELS
# ==========================

def gaussian_kernel_3x3():

    kernel = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ], dtype=np.float32)

    return kernel / 16


def gaussian_kernel_5x5():

    kernel = np.array([
        [1,4,6,4,1],
        [4,16,24,16,4],
        [6,24,36,24,6],
        [4,16,24,16,4],
        [1,4,6,4,1]
    ], dtype=np.float32)

    return kernel / np.sum(kernel)


# Simple approximation for 7x7
def gaussian_kernel_7x7():

    kernel = np.array([
        [1, 6,15,20,15, 6,1],
        [6,36,90,120,90,36,6],
        [15,90,225,300,225,90,15],
        [20,120,300,400,300,120,20],
        [15,90,225,300,225,90,15],
        [6,36,90,120,90,36,6],
        [1,6,15,20,15,6,1]
    ], dtype=np.float32)

    return kernel / np.sum(kernel)


# ==========================
# GAUSSIAN FILTER
# ==========================

def gaussian_filter(image, size):

    if size == 3:
        kernel = gaussian_kernel_3x3()

    elif size == 5:
        kernel = gaussian_kernel_5x5()

    else:
        kernel = gaussian_kernel_7x7()

    return convolution(image, kernel)


# ==========================
# MEDIAN FILTER
# ==========================

def median_filter(image, size):

    pad = size // 2

    padded = pad_image(image, pad)

    h, w = image.shape

    output = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    for i in range(h):

        for j in range(w):

            region = padded[
                i:i+size,
                j:j+size
            ]

            pixels = []

            for r in range(size):
                for c in range(size):

                    pixels.append(
                        region[r, c]
                    )

            pixels.sort()

            median = pixels[
                len(pixels)//2
            ]

            output[i, j] = median

    return output


# ==========================
# SALT PEPPER NOISE
# ==========================

def add_salt_pepper(image,
                    salt_prob=0.02,
                    pepper_prob=0.02):

    noisy = image.copy()

    h, w = image.shape

    total = h * w

    salt_pixels = int(total * salt_prob)
    pepper_pixels = int(total * pepper_prob)

    for _ in range(salt_pixels):

        x = np.random.randint(0, h)
        y = np.random.randint(0, w)

        noisy[x, y] = 255

    for _ in range(pepper_pixels):

        x = np.random.randint(0, h)
        y = np.random.randint(0, w)

        noisy[x, y] = 0

    return noisy


# ==========================
# MAIN PROCESS FUNCTION
# ==========================

def process_image(image, filter_name, kernel_size):

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    if filter_name == "Mean":
        result = mean_filter(
            gray,
            kernel_size
        )

    elif filter_name == "Box":
        result = box_filter(
            gray,
            kernel_size
        )

    elif filter_name == "Gaussian":
        result = gaussian_filter(
            gray,
            kernel_size
        )

    else:
        noisy = add_salt_pepper(gray)

        result = median_filter(
            noisy,
            kernel_size
        )

    return result


# ==========================
# EXAMPLES
# ==========================

examples = [
    ["examples/sample1.jpg", "Mean", 3],
    ["examples/sample2.jpg", "Gaussian", 5],
    ["examples/sample3.jpg", "Median", 3]
]


# ==========================
# GRADIO UI
# ==========================

demo = gr.Interface(
    fn=process_image,

    inputs=[
        gr.Image(label="Upload Image"),

        gr.Dropdown(
            choices=[
                "Mean",
                "Box",
                "Gaussian",
                "Median"
            ],
            value="Mean",
            label="Filter"
        ),

        gr.Radio(
            choices=[3,5,7],
            value=3,
            label="Kernel Size"
        )
    ],

    outputs=gr.Image(
        label="Filtered Output"
    ),

    title="Manual Image Filtering",

    description="""
    Mean Filter,
    Box Filter,
    Gaussian Filter,
    Median Filter

    Implemented Manually Using NumPy.
    No OpenCV Filtering Functions Used.
    """,

    examples=examples
)

demo.launch()