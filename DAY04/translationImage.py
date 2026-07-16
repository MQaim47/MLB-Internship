import cv2
import numpy as np
from PIL import Image

img = np.array(Image.open("H:\\Studies\\MLB-Internship\\DAY04\\image.jpg"))

rows, cols = img.shape[:2]

translated = np.zeros_like(img)

tx = 50
ty = 30

for y in range(rows):
    for x in range(cols):

        new_x = x + tx
        new_y = y + ty

        if new_x < cols and new_y < rows:
            translated[new_y, new_x] = img[y, x]
            
Image.fromarray(translated).save("translated.jpg")