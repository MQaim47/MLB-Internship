import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("image.jpg",0)

threshold_value=127

binary=np.zeros_like(img)
binary_inv=np.zeros_like(img)
truncated=np.zeros_like(img)
toZero=np.zeros_like(img)
toZero_inv=np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pixel=img[i,j]
        if pixel>threshold_value:
            binary[i,j]=255
        else:
            binary[i,j]=0
            
        # Binary Inverse
        if pixel>threshold_value:
            binary_inv[i,j]=0
        else:
            binary_inv[i,j]=255
            
        # Truncated
        if pixel>threshold_value:
            truncated[i,j]=threshold_value
        else:
            truncated[i,j]=pixel
            
        # To Zero
        if pixel>threshold_value:
            toZero[i,j]=pixel
        else:
            toZero[i,j]=0
        
        # To Zero Inverse
        if pixel>threshold_value:
            toZero_inv[i,j]=0
        else:
            toZero_inv[i,j]=pixel
            
cv2.imwrite("binary.jpg",binary)
cv2.imwrite("binary_inv.jpg",binary_inv)
cv2.imwrite("truncated.jpg",truncated)
cv2.imwrite("toZero.jpg",toZero)
cv2.imwrite("toZero_inv.jpg",toZero_inv)

print("Thresholding completed and images saved successfully.")
