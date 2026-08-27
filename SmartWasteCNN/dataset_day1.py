import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ---------------------------------
# Step 1: Image Transformations
# ---------------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ---------------------------------
# Step 2: Load Dataset
# ---------------------------------

dataset = datasets.ImageFolder(
    root="dataset/trashnet",   # Change to your dataset path
    transform=transform
)

# ---------------------------------
# Step 3: Dataset Information
# ---------------------------------

print("Classes:")
print(dataset.classes)

print("\nTotal Images:")
print(len(dataset))

# ---------------------------------
# Step 4: Split Dataset
# ---------------------------------

total_size = len(dataset)

train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print("\nTrain Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Test Images:", len(test_dataset))

# ---------------------------------
# Step 5: Create DataLoaders
# ---------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# ---------------------------------
# Step 6: Check Batch Shape
# ---------------------------------

images, labels = next(iter(train_loader))

print("\nImage Batch Shape:")
print(images.shape)

print("\nLabel Batch Shape:")
print(labels.shape)

# ---------------------------------
# Step 7: Display Sample Images
# ---------------------------------

plt.figure(figsize=(10, 6))

for i in range(6):

    img = images[i]

    # Convert C,H,W -> H,W,C
    img = img.permute(1, 2, 0)

    # Undo normalization
    img = (img * 0.5) + 0.5

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)

    plt.title(dataset.classes[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()