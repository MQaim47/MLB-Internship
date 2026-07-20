import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("image.jpg",0)

hist=np.zeros(256,dtype=int)

for row in img:
    for pixel in row:
        hist[pixel]+=1
        
plt.figure(figsize=(10,5))
plt.plot(hist)
plt.title("Histogram of the Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
plt.savefig("histogram.png")
