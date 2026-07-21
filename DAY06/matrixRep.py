from PIL import Image
import numpy as np

img=Image.open("image.jpg").convert("L")

img_array=np.array(img)

print("Image Matrix:")
print(img_array)


height, width = img_array.shape
print(f"Image Dimensions: {height} x {width}")

print("Pixel at (0, 0):", img_array[0, 0])
print("Pixel at (50, 50):", img_array[50, 50])

print("Pixel at (50, 100):", img_array[50, 100])

img_array[0][0] = 255
img_array[50][50] = 0
img_array[100][100] = 128

updated_img = Image.fromarray(img_array)
updated_img.save("updated_image.jpg")

