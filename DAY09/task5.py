import cv2
import numpy as np
import os

folder = "."

kernel = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])

threshold = 100

print(
    "\nImage\t\tScore\t\tStatus"
)

print("-"*50)

for file in os.listdir(folder):

    path = os.path.join(
        folder,
        file
    )

    img = cv2.imread(
        path,
        0
    )

    if img is None:
        continue

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

    if score < threshold:
        status = "Blurry"
    else:
        status = "Sharp"

    print(
        f"{file}\t{score:.2f}\t{status}"
    )
# import os
# import cv2
# import numpy as np
# import pandas as pd

# def task5_analysis():

#     folder = "test_images"

#     results = []

#     for file in os.listdir(folder):

#         path = os.path.join(
#             folder,
#             file
#         )

#         img = cv2.imread(
#             path,
#             0
#         )

#         if img is None:
#             continue

#         score = variance_of_laplacian(img)

#         if score > 100:
#             status = "Sharp"
#         else:
#             status = "Blurry"

#         observation = get_observation(score)

#         results.append([
#             file,
#             round(score,2),
#             status,
#             observation
#         ])

#     df = pd.DataFrame(
#         results,
#         columns=[
#             "Image Name",
#             "Variance Score",
#             "Blur Level",
#             "Observations"
#         ]
#     )

#     return df