from PIL import Image
img = Image.open("H:\\Studies\\MLB-Internship\\DAY02\\Image.jpg")

pixels=img.load()

pixels[10,10] = (255, 0, 0)
pixels[20,20] = (255, 0, 0)
pixels[30,30] = (255, 255, 255)
pixels[40,40] = (255, 255, 255)
pixels[50,50] = (255, 0, 0)  
img.save("H:\\Studies\\MLB-Internship\\DAY02\\Image_changed.jpg")

