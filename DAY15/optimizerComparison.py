import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------------
# Dataset (XOR)
# ----------------------------

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x




def train_model(optimizer_name):

    torch.manual_seed(42)

    model = NeuralNetwork()

    criterion = nn.MSELoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1
        )

    elif optimizer_name == "Momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9
        )

    elif optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.01
        )

    losses = []

    epochs = 2000

    for epoch in range(epochs):

        predictions = model(X)

        loss = criterion(predictions, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return losses




sgd_losses = train_model("SGD")

momentum_losses = train_model("Momentum")

adam_losses = train_model("Adam")



plt.figure(figsize=(10,6))

plt.plot(sgd_losses, label="SGD")

plt.plot(momentum_losses, label="SGD + Momentum")

plt.plot(adam_losses, label="Adam")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Optimizer Comparison")

plt.legend()

plt.grid(True)

plt.show()