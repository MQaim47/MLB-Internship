from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("image.jpg").convert("L")
img_array = np.array(img)

minimum = 255
maximum = 0
total = 0
count = 0

ranges = [0,0,0,0,0]

for row in img_array:
    for pixel in row:

        if pixel < minimum:
            minimum = pixel

        if pixel > maximum:
            maximum = pixel

        total += pixel
        count += 1

        if 0 <= pixel <= 50:
            ranges[0] += 1

        elif 51 <= pixel <= 100:
            ranges[1] += 1

        elif 101 <= pixel <= 150:
            ranges[2] += 1

        elif 151 <= pixel <= 200:
            ranges[3] += 1

        else:
            ranges[4] += 1

average = total / count

print("Minimum:",minimum)
print("Maximum:",maximum)
print("Average:",average)

print("\nIntensity Counts")
print("0-50 =",ranges[0])
print("51-100 =",ranges[1])
print("101-150 =",ranges[2])
print("151-200 =",ranges[3])
print("201-255 =",ranges[4])

plt.hist(img_array.ravel(), bins=256)
plt.title("Intensity Distribution")
plt.show()