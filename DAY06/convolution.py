from PIL import Image
import numpy as np

img = Image.open("image.jpg").convert("L")
img = np.array(img)

kernel = np.array([[1,1,1],
                [1,1,1],
                [1,1,1]])

kernel = kernel / 9

height, width = img.shape

output = np.zeros((height-2, width-2),dtype=np.uint8)

for i in range(height-2):
    for j in range(width-2):
        total = 0
        for k in range(3):
            for l in range(3):
                total += img[i+k][j+l] * kernel[k][l]
        output[i][j] = total

Image.fromarray(output).save("convolution_output.jpg")

print("Convolution Completed")


# kernel operation

identity = np.array([
    [0,0,0],
    [0,1,0],
    [0,0,0]
])

blur = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
])/9

sharpen = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

edge = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

def convolve(img,kernel):

    h,w = img.shape

    kh,kw = kernel.shape

    output = np.zeros((h-kh+1,w-kw+1))

    for i in range(h-kh+1):

        for j in range(w-kw+1):

            total = 0

            for ki in range(kh):

                for kj in range(kw):

                    total += img[i+ki][j+kj] * kernel[ki][kj]

            output[i][j] = total

    return np.clip(output,0,255).astype(np.uint8)

identity_img = convolve(img,identity)

blur_img = convolve(img,blur)

sharpen_img = convolve(img,sharpen)

edge_img = convolve(img,edge)