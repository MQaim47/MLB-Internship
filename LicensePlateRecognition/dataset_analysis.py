import os

train_images = len(os.listdir("dataset/train/images"))
valid_images = len(os.listdir("dataset/valid/images"))
test_images = len(os.listdir("dataset/test/images"))

print("Train:", train_images)
print("Valid:", valid_images)
print("Test :", test_images)

print("Total Images:", train_images + valid_images + test_images)