from PIL import Image
img=Image.open("H:\Studies\MLB-Internship\DAY02\Image.jpg")
width, height = img.size
pixel_values = list(img.getdata())
channels=len(pixel_values[0])
print("Height:", height)
print("Width:", width)
print("Channels:", channels)
print("Data Type:", img.mode)

total_pixels = width * height
print("Total Pixels:", total_pixels)