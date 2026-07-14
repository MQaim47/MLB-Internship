from PIL import Image
img = Image.open("H:\\Studies\\MLB-Internship\\DAY02\\image2.jpg")

width, height = img.size

r_img=Image.new('L', (width, height))
b_img=Image.new('L', (width, height))
g_img=Image.new('L', (width, height))

pixels=img.load()

for y in range(height):
    for x in range(width):
        r, g, b = pixels[x, y]
        r_img.putpixel((x, y), r)
        g_img.putpixel((x, y), g)
        b_img.putpixel((x, y), b)
        
r_img.save("H:\\Studies\\MLB-Internship\\DAY02\\R_Image.jpg")
g_img.save("H:\\Studies\\MLB-Internship\\DAY02\\G_Image.jpg")
b_img.save("H:\\Studies\\MLB-Internship\\DAY02\\B_Image.jpg")