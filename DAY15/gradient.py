import numpy as np

X = np.array([1, 2, 3, 4, 5])

y = np.array([3, 5, 7, 9, 11])


w = 0.0
b = 0.0


learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):

    y_pred = w * X + b

    loss = np.mean((y - y_pred) ** 2)

    dw = (-2/n) * np.sum(X * (y - y_pred))

    db = (-2/n) * np.sum(y - y_pred)

    w = w - learning_rate * dw

    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | "
            f"Loss = {loss:.4f}"
        )

print("\nFinal Weight:", w)
print("Final Bias:", b)