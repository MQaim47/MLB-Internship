from PIL import Image
import numpy as np

img = Image.open("image.jpg").convert("L")
img_array = np.array(img)

coordinates = [
    (10,10),
    (50,80),
    (100,150),
    (200,250)
]

for y,x in coordinates:
    print("Coordinate:",(x,y))
    print("Pixel Value:",img_array[y][x])
    print()