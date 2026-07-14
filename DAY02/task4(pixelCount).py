from PIL import Image
img = Image.open("H:\\Studies\\MLB-Internship\\DAY02\\Image.jpg")

pixels=list(img.getdata())

total_pixels=0
black_pixels=0
white_pixels=0
above_200_pixels=0

for pixel in pixels:
    total_pixels+=1
    if pixel==(0,0,0):
        black_pixels+=1
    elif pixel==(255,255,255):
        white_pixels+=1
    elif pixel[0]>200 and pixel[1]>200 and pixel[2]>200:
        above_200_pixels+=1

print("Total Pixels:", total_pixels)
print("Black Pixels:", black_pixels)
print("White Pixels:", white_pixels)
print("Pixels with all channels above 200:", above_200_pixels)