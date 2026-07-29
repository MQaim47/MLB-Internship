import cv2
import numpy as np

img = cv2.imread(
    "input.jpg",
    0
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

print("Blur Score =",score)

threshold = 100

if score < threshold:
    print("Blurry")
else:
    print("Sharp")