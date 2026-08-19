import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

x=np.linspace(-10, 10, 100)

sigmoid_y = sigmoid(x)
tanh_y = tanh(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)

softmax_input = np.array([1.0, 2.0, 3.0])
softmax_output = softmax(softmax_input)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_y)
plt.title("Sigmoid")

plt.subplot(2, 2, 2)
plt.plot(x, tanh_y)
plt.title("Tanh")

plt.subplot(2, 2, 3)
plt.plot(x, relu_y)
plt.title("ReLU")

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu_y)
plt.title("Leaky ReLU")

plt.tight_layout()
plt.show()

print("Softmax Input :", softmax_input)
print("Softmax Output:", softmax_output)