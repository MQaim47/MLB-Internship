import torch
import torch.nn as nn
import torch.optim as optim
from dataset_day1 import train_loader

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


model = SimpleCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 10

for epoch in range(num_epochs):

    running_loss = 0.0

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

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

torch.save(
    model.state_dict(),
    "waste_classifier.pth"
)