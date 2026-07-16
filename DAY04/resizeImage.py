from PIL import Image
import numpy as np

import cv2

img = np.array(Image.open("H:\\Studies\\MLB-Internship\\DAY04\\image.jpg"))



new_h = 300
new_w = 400

resized = np.zeros((new_h,new_w,3),dtype=img.dtype)

for y in range(new_h):
    for x in range(new_w):

        src_y = int(y * img.shape[0] / new_h)
        src_x = int(x * img.shape[1] / new_w)

        resized[y,x] = img[src_y,src_x]
        
Image.fromarray(resized).save("H:\\Studies\\MLB-Internship\\DAY04\\resized.jpg")