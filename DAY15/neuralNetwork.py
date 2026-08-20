import torch
import torch.nn as nn
import torch.optim as optim



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

        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.hidden(x)
        x = self.sigmoid(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x



model = NeuralNetwork()



criterion = nn.MSELoss()


optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)


epochs = 10000

for epoch in range(epochs):

    predictions = model(X)

    loss = criterion(predictions, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss.item():.6f}"
        )


with torch.no_grad():

    predictions = model(X)

    print("\nFinal Predictions:")
    print(predictions)