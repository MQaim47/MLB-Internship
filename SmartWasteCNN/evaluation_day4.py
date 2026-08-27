import torch
import torch.nn as nn

from torchvision import datasets, transforms

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

# Transform

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])

# Dataset

dataset = datasets.ImageFolder(
    root="dataset/trashnet",
    transform=transform
)

classes = dataset.classes

# Split Dataset

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# CNN Model

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3,16,3)

        self.conv2 = nn.Conv2d(16,32,3)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(
            32*30*30,
            128
        )

        self.fc2 = nn.Linear(
            128,
            6
        )

    def forward(self,x):

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

        x = torch.flatten(x,1)

        x = self.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x

# Load Model

model = SimpleCNN()

model.load_state_dict(
    torch.load(
        "waste_classifier_day3.pth"
    )
)

model.eval()

print("Model Loaded Successfully!")

# Evaluation

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predicted.numpy()
        )

# Metrics

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print("\nAccuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

# Detailed Report

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )
)


# Confusion Matrix


cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()