import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Dataset
X = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0],
                  [5.0]])

y = torch.tensor([[2.0],
                  [4.0],
                  [6.0],
                  [8.0],
                  [10.0]])

# Neural Network
model = nn.Sequential(
    nn.Linear(1, 1)
)

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []

epochs = 100

for epoch in range(epochs):

    prediction = model(X)

    loss = criterion(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}, Loss = {loss.item():.4f}"
        )

print("\nPredictions:")

with torch.no_grad():
    print(model(X))

# Plot Loss
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.grid(True)
plt.show()