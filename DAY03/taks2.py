import cv2
import matplotlib.pyplot as plt
import numpy as np

img =cv2.imread("H:\\Studies\\MLB-Internship\\DAY03\\image.jpg")
img =img [:,:,::-1]

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

# merged = cv2.merge([R,G,B])

plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(R,cmap="gray")
plt.title("Red Channel")

plt.subplot(1,3,2)
plt.imshow(G,cmap="gray")
plt.title("Green Channel")

plt.subplot(1,3,3)
plt.imshow(B,cmap="gray")
plt.title("Blue Channel")
plt.show()

merged =np.zeros_like(img)
merged[:,:,0] = R
merged[:,:,1] = G
merged[:,:,2] = B
plt.show()
