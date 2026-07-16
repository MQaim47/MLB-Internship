from PIL import Image
import numpy as np

import cv2

img = np.array(Image.open("H:\\Studies\\MLB-Internship\\DAY04\\image.jpg"))


factor = 2

h,w,c = img.shape

new_h = int(h*factor)
new_w = int(w*factor)

scaled = np.zeros((new_h,new_w,c),dtype=img.dtype)

for y in range(new_h):
    for x in range(new_w):

        old_y = int(y/factor)
        old_x = int(x/factor)

        scaled[y,x] = img[old_y,old_x]
        
Image.fromarray(scaled).save("H:\\Studies\\MLB-Internship\\DAY04\\scaled.jpg")