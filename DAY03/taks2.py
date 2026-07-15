import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

merged = cv2.merge([R,G,B])

plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(R,cmap="gray")
plt.title("Red Channel")

plt.subplot(2,2,2)
plt.imshow(G,cmap="gray")
plt.title("Green Channel")

plt.subplot(2,2,3)
plt.imshow(B,cmap="gray")
plt.title("Blue Channel")

plt.subplot(2,2,4)
plt.imshow(merged)
plt.title("Merged Image")

plt.show()