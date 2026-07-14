from PIL import Image
img = Image.open("H:\\Studies\\MLB-Internship\\DAY02\\Image.jpg")
img = img.convert('L')  # Convert to grayscale
pixels = list(img.getdata())

total=0
count=0

min_pixel = 255
max_pixel = 0

for pixel in pixels:
    total += pixel
    count += 1
    
    if pixel<min_pixel:
        min_pixel = pixel
    if pixel>max_pixel:
        max_pixel = pixel

mean=total/count

print("Mean:", mean)
print("Min:", min_pixel)
print("Max:", max_pixel)

