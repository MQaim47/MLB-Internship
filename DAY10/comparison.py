import cv2
import numpy as np
import matplotlib.pyplot as plt

# Same image
image = cv2.imread("image.jpg", 0)

# Sobel
_, _, sobel_result = sobel_edge_detection(image)

# Laplacian
laplacian_result = laplacian_edge_detection(image)

# LoG
_, log_result = log_edge_detection(image)

# DoG
_, _, dog_result = dog_edge_detection(image)

# Canny
_, _, _, _, canny_result = manual_canny(image)

# Normalize
sobel_result = normalize_image(sobel_result)
laplacian_result = normalize_image(laplacian_result)
log_result = normalize_image(log_result)
dog_result = normalize_image(dog_result)

# Display
plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(sobel_result, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(laplacian_result, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(log_result, cmap="gray")
plt.title("LoG")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(dog_result, cmap="gray")
plt.title("DoG")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(canny_result, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()