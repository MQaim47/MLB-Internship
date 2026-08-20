import matplotlib.pyplot as plt

from PIL import Image

from torchvision import transforms



image = Image.open("cat.jpg")



augmentations = {

    "Original": transforms.Compose([]),

    "Horizontal Flip":
        transforms.RandomHorizontalFlip(p=1),

    "Vertical Flip":
        transforms.RandomVerticalFlip(p=1),

    "Rotation":
        transforms.RandomRotation(45),

    "Crop":
        transforms.RandomCrop(
            (100, 100)
        ),

    "Resize":
        transforms.Resize(
            (128, 128)
        ),

    "Brightness":
        transforms.ColorJitter(
            brightness=0.8
        ),

    "Contrast":
        transforms.ColorJitter(
            contrast=0.8
        )
}



plt.figure(figsize=(12,8))

for i, (name, transform) in enumerate(
        augmentations.items(), 1):

    augmented = transform(image)

    plt.subplot(2,4,i)

    plt.imshow(augmented)

    plt.title(name)

    plt.axis("off")

plt.tight_layout()

plt.show()