from PIL import Image
img = Image.open("H:\\Studies\\MLB-Internship\\DAY02\\image2.jpg")

width, height = img.size

grayscale_img = Image.new('L', (width, height))
pixels = img.load()

for y in range(height):
    for x in range(width):
        r, g, b = pixels[x, y]
        gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
        grayscale_img.putpixel((x, y), gray_value)

grayscale_img.save("H:\\Studies\\MLB-Internship\\DAY02\\Grayscale_Image.jpg")