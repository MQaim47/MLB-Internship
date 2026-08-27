import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

# Classes

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# CNN

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

# Transform

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])

# Load Image


image = Image.open(
    "trash1.jpg"
).convert("RGB")

image = transform(
    image
).unsqueeze(0)

# Predict

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

print(
    "Predicted Class:",
    classes[predicted.item()]
)

print(
    "Confidence:",
    confidence.item()*100,
    "%"
)