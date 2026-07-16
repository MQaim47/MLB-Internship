from PIL import Image
import numpy as np

import cv2

img = np.array(Image.open("H:\\Studies\\MLB-Internship\\DAY04\\image.jpg"))

padded = np.pad(
    img,
    ((50,50),(50,50),(0,0)),
    mode='constant',
    constant_values=0
)
Image.fromarray(padded).save("H:\\Studies\\MLB-Internship\\DAY04\\padded.jpg")

# normalization
normalized = img.astype(float)/255.0
print("Minimum Value:", normalized.min())
print("Maximum Value:", normalized.max())
# Convert back for saving
normalized_save = (normalized * 255).astype(np.uint8)
Image.fromarray(normalized_save).save("H:\\Studies\\MLB-Internship\\DAY04\\normalized.jpg")