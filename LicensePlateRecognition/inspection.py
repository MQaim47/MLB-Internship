import cv2
import os
import random

path = "dataset/train/images"

images = os.listdir(path)

for i in range(20):
    img_name = random.choice(images)

    img = cv2.imread(os.path.join(path,img_name))

    cv2.imshow("Image",img)
    cv2.waitKey(500)

cv2.destroyAllWindows()