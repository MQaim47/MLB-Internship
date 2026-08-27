import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

# Data Augmentation

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# Load Dataset


dataset = datasets.ImageFolder(
    root="dataset/trashnet",
    transform=train_transform
)

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

# validation dataset transform

val_dataset.dataset.transform = val_transform

# Data Loaders

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

# CNN Model

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            3
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            3
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.fc1 = nn.Linear(
            32 * 30 * 30,
            128
        )

        self.fc2 = nn.Linear(
            128,
            6
        )

    def forward(self, x):

        x = self.pool(
            self.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            self.relu(
                self.conv2(x)
            )
        )

        x = torch.flatten(x, 1)

        x = self.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x

# Create Model

model = SimpleCNN()

# Loss Function

criterion = nn.CrossEntropyLoss()

# Optimizer


optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Store Metrics

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

# Training

num_epochs = 10

for epoch in range(num_epochs):

    # TRAINING

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )

    # VALIDATION

    model.eval()

    val_running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_loss = (
        val_running_loss /
        len(val_loader)
    )

    val_accuracy = (
        100 * correct / total
    )

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )

# Save Model

torch.save(
    model.state_dict(),
    "waste_classifier_day3.pth"
)

print("\nModel Saved!")

# Loss Graph

plt.figure(figsize=(8,5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()

# Accuracy Graph


plt.figure(figsize=(8,5))

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.legend()

plt.grid()

plt.show()