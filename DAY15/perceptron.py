import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        print("Initial Weights:", self.weights)
        print("Initial Bias:", self.bias)
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}")

            for i in range(len(X)):
                z = np.dot(X[i], self.weights) + self.bias

                y_pred = self.activation(z)

                error = y[i] - y_pred

                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

                print(f"Input: {X[i]}")
                print(f"Actual: {y[i]}")
                print(f"Predicted: {y_pred}")
                print(f"Error: {error}")
                print(f"Weights: {self.weights}")
                print(f"Bias: {self.bias}")

    def predict(self, X):
        predictions = []

        for x in X:
            z = np.dot(x, self.weights) + self.bias
            predictions.append(self.activation(z))

        return np.array(predictions)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

model = Perceptron(
    learning_rate=0.1,
    epochs=10
)

model.fit(X, y)

predictions = model.predict(X)

print("\nFinal Weights:", model.weights)
print("Final Bias:", model.bias)

print("\nPredictions:")
for inp, pred in zip(X, predictions):
    print(inp, "->", pred)