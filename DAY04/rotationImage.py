from PIL import Image
import numpy as np

import cv2

img = np.array(Image.open("H:\\Studies\\MLB-Internship\\DAY04\\image.jpg"))
h,w,c = img.shape

rotated = np.zeros((w,h,c),dtype=img.dtype)

for y in range(h):
    for x in range(w):

        rotated[x,h-1-y] = img[y,x]
        
Image.fromarray(rotated).save("H:\\Studies\\MLB-Internship\\DAY04\\rotated.jpg")