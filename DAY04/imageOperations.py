from PIL import Image
import numpy as np

img = np.array(Image.open("H:\\Studies\\MLB-Internship\\DAY04\\image.jpg"))

import cv2
# read image
img = cv2.imread("H:\\Studies\\MLB-Internship\\DAY04\\image.jpg")
print(img.shape)
# save image

cv2.imwrite("H:\\Studies\\MLB-Internship\\DAY04\\saved_image.jpg", img)
# Copy image
copy_img = img.copy()
cv2.imwrite("H:\\Studies\\MLB-Internship\\DAY04\\copy.jpg", copy_img)

# cropped image
cropped = img[100:300, 150:400]
cv2.imwrite("H:\\Studies\\MLB-Internship\\DAY04\\cropped.jpg", cropped)

# flipped image
flip_h=img[:,::-1]
cv2.imwrite("H:\\Studies\\MLB-Internship\\DAY04\\flip_h.jpg", flip_h)
flip_v = img[::-1, :]
cv2.imwrite("H:\\Studies\\MLB-Internship\\DAY04\\flip_v.jpg", flip_v)
flip_both = img[::-1, ::-1]
