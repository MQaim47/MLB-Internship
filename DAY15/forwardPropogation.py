import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([[1, 2]])

W1 = np.array([
    [0.5, 0.2],
    [0.3, 0.8]
])

b1 = np.array([[0.1, 0.2]])

W2 = np.array([
    [0.4],
    [0.7]
])

b2 = np.array([[0.3]])

Z1 = np.dot(X, W1) + b1
A1 = sigmoid(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)


print("Input:")
print(X)

print("\nHidden Layer Weighted Sum (Z1):")
print(Z1)

print("\nHidden Layer Activation (A1):")
print(A1)

print("\nOutput Layer Weighted Sum (Z2):")
print(Z2)

print("\nFinal Output (A2):")
print(A2)