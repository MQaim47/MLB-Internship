"""
DAY 15 - Neural Networks & Deep Learning Fundamentals
Single-file multi-tab Gradio app.

Combines all DAY15 tasks (perceptron, forward/back propagation, activation &
loss functions, gradient descent, a PyTorch neural net, optimizer comparison,
data augmentation, transfer learning) plus the mini project (transfer
learning + fine-tuning on a cat/dog dataset) into one Gradio interface.

Run with:
    python app.py
"""

import io
import os
import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from PIL import Image

import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAT_IMAGE_PATH = os.path.join(BASE_DIR, "cat.jpg")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# TASK 1 - PERCEPTRON (AND Gate)
# ======================================================================

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y, log_fn=None):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        if log_fn:
            log_fn(f"Initial Weights: {self.weights}, Initial Bias: {self.bias}\n")

        for epoch in range(self.epochs):
            epoch_lines = [f"--- Epoch {epoch + 1} ---"]
            for i in range(len(X)):
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(z)
                error = y[i] - y_pred

                self.weights = self.weights + self.learning_rate * error * X[i]
                self.bias = self.bias + self.learning_rate * error

                epoch_lines.append(
                    f"Input: {X[i]} | Actual: {y[i]} | Predicted: {y_pred} | "
                    f"Error: {error} | Weights: {np.round(self.weights, 3)} | "
                    f"Bias: {round(self.bias, 3)}"
                )
            if log_fn:
                log_fn("\n".join(epoch_lines) + "\n")

    def predict(self, X):
        predictions = []
        for x in X:
            z = np.dot(x, self.weights) + self.bias
            predictions.append(self.activation(z))
        return np.array(predictions)


def run_perceptron(learning_rate, epochs):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    logs = []
    model = Perceptron(learning_rate=learning_rate, epochs=int(epochs))
    model.fit(X, y, log_fn=lambda s: logs.append(s))

    predictions = model.predict(X)

    result_lines = ["\nFinal Weights: " + str(np.round(model.weights, 4)),
                     "Final Bias: " + str(round(model.bias, 4)),
                     "\nPredictions (AND gate):"]
    for inp, pred, actual in zip(X, predictions, y):
        mark = "OK" if pred == actual else "X"
        result_lines.append(f"  {inp} -> predicted={pred}  actual={actual}  [{mark}]")

    full_log = "\n".join(logs) + "\n" + "\n".join(result_lines)

    # Decision boundary plot
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["red" if label == 0 else "green" for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors="black", zorder=3)
    for (x0, x1), label in zip(X, y):
        ax.annotate(f"({x0},{x1})->{label}", (x0, x1), textcoords="offset points",
                    xytext=(10, 10))

    w1, w2 = model.weights
    b = model.bias
    xs = np.linspace(-0.5, 1.5, 100)
    if abs(w2) > 1e-8:
        ys = -(w1 * xs + b) / w2
        ax.plot(xs, ys, "b--", label="Decision boundary")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Perceptron Decision Boundary (AND gate)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)

    return full_log, fig


# ======================================================================
# TASK 2 - FORWARD PROPAGATION
# ======================================================================

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def run_forward_propagation(x1, x2):
    X = np.array([[x1, x2]])

    W1 = np.array([[0.5, 0.2], [0.3, 0.8]])
    b1 = np.array([[0.1, 0.2]])

    W2 = np.array([[0.4], [0.7]])
    b2 = np.array([[0.3]])

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid_np(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid_np(Z2)

    out = io.StringIO()
    out.write(f"Input X:\n{X}\n\n")
    out.write("Fixed demo weights: W1, b1 (hidden layer), W2, b2 (output layer)\n\n")
    out.write(f"Hidden layer weighted sum (Z1 = X.W1 + b1):\n{np.round(Z1, 5)}\n\n")
    out.write(f"Hidden layer activation (A1 = sigmoid(Z1)):\n{np.round(A1, 5)}\n\n")
    out.write(f"Output layer weighted sum (Z2 = A1.W2 + b2):\n{np.round(Z2, 5)}\n\n")
    out.write(f"Final output (A2 = sigmoid(Z2)):\n{np.round(A2, 5)}\n")

    return out.getvalue()


# ======================================================================
# TASK 3 - BACKPROPAGATION (manual NumPy, XOR)
# ======================================================================

def run_backpropagation(learning_rate, epochs):
    epochs = int(epochs)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(2, 2)
    b1 = np.zeros((1, 2))
    W2 = np.random.randn(2, 1)
    b2 = np.zeros((1, 1))

    losses = []
    log_lines = []

    for epoch in range(epochs):
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        y_pred = sigmoid(Z2)

        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)

        dZ2 = (y_pred - y) * sigmoid_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * sigmoid_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch % max(1, epochs // 10) == 0:
            log_lines.append(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

    log_lines.append(f"Epoch {epochs - 1:5d} | Loss: {losses[-1]:.6f}")
    log_lines.append("\nFinal Predictions (XOR):")
    for inp, target, pred in zip(X, y, y_pred):
        log_lines.append(f"  {inp} -> target={target[0]}  predicted={pred[0]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Manual Backpropagation - Loss Curve (XOR)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)

    return "\n".join(log_lines), fig


# ======================================================================
# TASK 4 - ACTIVATION FUNCTIONS
# ======================================================================

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


def run_activation_functions(softmax_input_str):
    x = np.linspace(-10, 10, 100)

    sigmoid_y = sigmoid(x)
    tanh_y = tanh(x)
    relu_y = relu(x)
    leaky_relu_y = leaky_relu(x)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(x, sigmoid_y, color="tab:blue")
    axes[0, 0].set_title("Sigmoid")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, tanh_y, color="tab:orange")
    axes[0, 1].set_title("Tanh")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x, relu_y, color="tab:green")
    axes[1, 0].set_title("ReLU")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, leaky_relu_y, color="tab:red")
    axes[1, 1].set_title("Leaky ReLU")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.close(fig)

    try:
        values = [float(v.strip()) for v in softmax_input_str.split(",") if v.strip() != ""]
        if len(values) == 0:
            raise ValueError("empty input")
        softmax_input = np.array(values)
        softmax_output = softmax(softmax_input)
        text = (f"Softmax Input : {softmax_input}\n"
                f"Softmax Output: {np.round(softmax_output, 5)}\n"
                f"Sum of outputs: {softmax_output.sum():.5f}")
    except Exception as e:
        text = f"Could not parse softmax input ({e}). Use comma-separated numbers, e.g. 1.0, 2.0, 3.0"

    return fig, text


# ======================================================================
# TASK 5 - LOSS FUNCTIONS
# ======================================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))


LOSS_DEFAULTS = {
    "MSE": ("1, 0, 1", "0.8, 0.2, 0.9"),
    "Binary Cross-Entropy": ("1, 0, 1, 0", "0.9, 0.2, 0.8, 0.1"),
    "Categorical Cross-Entropy": ("1, 0, 0", "0.8, 0.1, 0.1"),
}


def loss_type_change(loss_type):
    y_true_default, y_pred_default = LOSS_DEFAULTS[loss_type]
    return y_true_default, y_pred_default


def run_loss_function(loss_type, y_true_str, y_pred_str):
    try:
        y_true = np.array([float(v.strip()) for v in y_true_str.split(",") if v.strip() != ""])
        y_pred = np.array([float(v.strip()) for v in y_pred_str.split(",") if v.strip() != ""])
        if len(y_true) != len(y_pred):
            return "y_true and y_pred must have the same number of values."

        if loss_type == "MSE":
            value = mse(y_true, y_pred)
            formula = "MSE = mean((y_true - y_pred)^2)"
        elif loss_type == "Binary Cross-Entropy":
            value = binary_cross_entropy(y_true, y_pred)
            formula = "BCE = -mean(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))"
        else:
            value = categorical_cross_entropy(y_true, y_pred)
            formula = "CCE = -sum(y_true*log(y_pred))"

        return (f"{loss_type}\n{formula}\n\n"
                f"y_true = {y_true}\ny_pred = {y_pred}\n\n"
                f"Loss = {value:.6f}")
    except Exception as e:
        return f"Error parsing input: {e}. Use comma-separated numbers."


# ======================================================================
# TASK 6 - GRADIENT DESCENT (Linear Regression)
# ======================================================================

def run_gradient_descent(x_str, y_str, learning_rate, epochs):
    epochs = int(epochs)
    try:
        X = np.array([float(v.strip()) for v in x_str.split(",") if v.strip() != ""])
        y = np.array([float(v.strip()) for v in y_str.split(",") if v.strip() != ""])
        if len(X) != len(y) or len(X) == 0:
            return "X and y must be non-empty and the same length.", None
    except Exception as e:
        return f"Error parsing input: {e}", None

    w, b = 0.0, 0.0
    n = len(X)
    losses = []
    log_lines = []

    for epoch in range(epochs):
        y_pred = w * X + b
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)

        dw = (-2 / n) * np.sum(X * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if epoch % max(1, epochs // 10) == 0:
            log_lines.append(f"Epoch {epoch:5d} | Loss = {loss:.4f}")

    log_lines.append(f"\nFinal Weight (w): {w:.4f}")
    log_lines.append(f"Final Bias (b): {b:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(losses)
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(X, y, color="tab:blue", label="Data", zorder=3)
    xs = np.linspace(X.min() - 1, X.max() + 1, 50)
    ax2.plot(xs, w * xs + b, color="tab:red", label=f"Fit: y = {w:.2f}x + {b:.2f}")
    ax2.set_title("Fitted line")
    ax2.set_xlabel("X")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)

    return "\n".join(log_lines), fig


# ======================================================================
# TASK 7 - NEURAL NETWORK (PyTorch, XOR)
# ======================================================================

class XORNet(nn.Module):
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


def run_neural_network(learning_rate, epochs):
    epochs = int(epochs)
    torch.manual_seed(42)

    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])

    model = XORNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    log_lines = []

    for epoch in range(epochs):
        predictions = model(X)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % max(1, epochs // 10) == 0:
            log_lines.append(f"Epoch {epoch:5d}, Loss: {loss.item():.6f}")

    with torch.no_grad():
        predictions = model(X)

    log_lines.append(f"\nFinal Loss: {losses[-1]:.6f}")
    log_lines.append("\nFinal Predictions (XOR):")
    for inp, target, pred in zip(X, y, predictions):
        log_lines.append(f"  {inp.tolist()} -> target={target.item():.0f}  predicted={pred.item():.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("PyTorch Neural Network - Loss Curve (XOR)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)

    return "\n".join(log_lines), fig


# ======================================================================
# TASK 8 - OPTIMIZER COMPARISON
# ======================================================================

class OptimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def _train_optim_model(optimizer_name, X, y, epochs):
    torch.manual_seed(42)
    model = OptimNet()
    criterion = nn.MSELoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    else:  # Adam
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(epochs):
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def run_optimizer_comparison(epochs):
    epochs = int(epochs)
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])

    sgd_losses = _train_optim_model("SGD", X, y, epochs)
    momentum_losses = _train_optim_model("Momentum", X, y, epochs)
    adam_losses = _train_optim_model("Adam", X, y, epochs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sgd_losses, label="SGD")
    ax.plot(momentum_losses, label="SGD + Momentum")
    ax.plot(adam_losses, label="Adam")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Optimizer Comparison (XOR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)

    summary = (f"Final Loss after {epochs} epochs:\n"
               f"  SGD             : {sgd_losses[-1]:.6f}\n"
               f"  SGD + Momentum  : {momentum_losses[-1]:.6f}\n"
               f"  Adam            : {adam_losses[-1]:.6f}")

    return fig, summary


# ======================================================================
# TASK 9 - DATA AUGMENTATION
# ======================================================================

def run_data_augmentation(image):
    if image is None:
        if os.path.exists(CAT_IMAGE_PATH):
            image = Image.open(CAT_IMAGE_PATH).convert("RGB")
        else:
            return []
    else:
        image = image.convert("RGB")

    augmentations = {
        "Original": transforms.Compose([]),
        "Horizontal Flip": transforms.RandomHorizontalFlip(p=1),
        "Vertical Flip": transforms.RandomVerticalFlip(p=1),
        "Rotation (45deg)": transforms.RandomRotation(45),
        "Crop (100x100)": transforms.RandomCrop((100, 100)) if min(image.size) >= 100
        else transforms.CenterCrop(min(image.size)),
        "Resize (128x128)": transforms.Resize((128, 128)),
        "Brightness": transforms.ColorJitter(brightness=0.8),
        "Contrast": transforms.ColorJitter(contrast=0.8),
    }

    gallery = []
    for name, transform in augmentations.items():
        augmented = transform(image)
        gallery.append((augmented, name))

    return gallery


# ======================================================================
# TASK 10 - TRANSFER LEARNING (frozen ResNet18 backbone)
# ======================================================================

def _build_dataloaders(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_transform)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader


def _load_pretrained_resnet18(log_lines):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        log_lines.append("Loaded ImageNet-pretrained ResNet18 weights.")
    except Exception as e:
        log_lines.append(f"Could not download pretrained weights ({e}).")
        log_lines.append("Falling back to randomly-initialized ResNet18 "
                          "(needs internet access on first run to fetch ImageNet weights).")
        model = models.resnet18(weights=None)
    return model


def run_transfer_learning(epochs):
    epochs = int(epochs)
    if not os.path.isdir(DATASET_DIR):
        return "dataset/ folder not found next to app.py."

    log_lines = []
    train_dataset, val_dataset, train_loader, val_loader = _build_dataloaders()

    log_lines.append(f"Using device: {DEVICE}")
    log_lines.append(f"Classes: {train_dataset.classes}")
    log_lines.append(f"Training images: {len(train_dataset)}")
    log_lines.append(f"Validation images: {len(val_dataset)}\n")

    model = _load_pretrained_resnet18(log_lines)

    for param in model.parameters():
        param.requires_grad = False

    number_of_classes = len(train_dataset.classes)
    model.fc = nn.Linear(512, number_of_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    log_lines.append("\nTraining only the final layer (frozen backbone):")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        log_lines.append(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total if total > 0 else 0.0
    log_lines.append(f"\nValidation Accuracy: {val_accuracy:.2f}%")
    log_lines.append("\n(Note: this demo dataset only has a handful of images per "
                      "class, so accuracy numbers are illustrative, not meaningful.)")

    return "\n".join(log_lines)


# ======================================================================
# MINI PROJECT (task10) - Transfer Learning + Fine-Tuning + live predict
# ======================================================================

_mini_project_state = {"model": None, "classes": None, "transform": None}


def run_mini_project(transfer_epochs, finetune_epochs):
    transfer_epochs = int(transfer_epochs)
    finetune_epochs = int(finetune_epochs)

    if not os.path.isdir(DATASET_DIR):
        return "dataset/ folder not found next to app.py.", gr.update(interactive=False)

    log_lines = []
    train_dataset, val_dataset, train_loader, val_loader = _build_dataloaders()
    classes = train_dataset.classes
    number_of_classes = len(classes)

    log_lines.append(f"Using device: {DEVICE}")
    log_lines.append(f"Classes: {classes}")
    log_lines.append(f"Training images: {len(train_dataset)}")
    log_lines.append(f"Validation images: {len(val_dataset)}")

    model = _load_pretrained_resnet18(log_lines)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, number_of_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    log_lines.append("\n=== PHASE 1: TRANSFER LEARNING (frozen backbone) ===")
    for epoch in range(transfer_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        log_lines.append(f"Epoch [{epoch + 1}/{transfer_epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")

    log_lines.append("\n=== PHASE 2: FINE-TUNING (unfreeze layer4) ===")
    for param in model.layer4.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    for epoch in range(finetune_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        log_lines.append(f"Fine-Tune Epoch [{epoch + 1}/{finetune_epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total if total > 0 else 0.0
    log_lines.append(f"\nFinal Validation Accuracy: {val_accuracy:.2f}%")
    log_lines.append("\n(Note: this demo dataset only has a handful of images per "
                      "class, so accuracy numbers are illustrative, not meaningful. "
                      "You can now try the 'Predict on your own image' box below.)")

    predict_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _mini_project_state["model"] = model
    _mini_project_state["classes"] = classes
    _mini_project_state["transform"] = predict_transform

    return "\n".join(log_lines), gr.update(interactive=True)


def predict_mini_project(image):
    if _mini_project_state["model"] is None:
        return "Run the training pipeline above first, then try a prediction."
    if image is None:
        return "Upload an image first."

    model = _mini_project_state["model"]
    classes = _mini_project_state["classes"]
    transform = _mini_project_state["transform"]

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    lines = [f"Predicted class: {classes[predicted.item()]}", "", "Class probabilities:"]
    for cls, p in zip(classes, probs.tolist()):
        lines.append(f"  {cls}: {p * 100:.2f}%")

    return "\n".join(lines)


# ======================================================================
# GRADIO APP
# ======================================================================

with gr.Blocks(title="DAY 15 - Neural Networks & Deep Learning") as demo:
    gr.Markdown("# DAY 15 - Neural Networks & Deep Learning Fundamentals")
    gr.Markdown(
        "Interactive versions of every DAY15 task, plus the mini project "
        "(transfer learning + fine-tuning on a cat/dog dataset), all in one app."
    )

    with gr.Tabs():

        # ---------------- Tab 1: Perceptron ----------------
        with gr.Tab("1. Perceptron"):
            gr.Markdown("### Perceptron learning the AND gate")
            with gr.Row():
                lr_p = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Learning Rate")
                epochs_p = gr.Slider(1, 50, value=10, step=1, label="Epochs")
            btn_p = gr.Button("Train Perceptron", variant="primary")
            with gr.Row():
                log_p = gr.Textbox(label="Training Log", lines=20, max_lines=30)
                plot_p = gr.Plot(label="Decision Boundary")
            btn_p.click(run_perceptron, inputs=[lr_p, epochs_p], outputs=[log_p, plot_p])

        # ---------------- Tab 2: Forward Propagation ----------------
        with gr.Tab("2. Forward Propagation"):
            gr.Markdown("### Manual forward pass through a 2-2-1 network (fixed demo weights)")
            with gr.Row():
                x1_fp = gr.Number(value=1, label="x1")
                x2_fp = gr.Number(value=2, label="x2")
            btn_fp = gr.Button("Run Forward Pass", variant="primary")
            out_fp = gr.Textbox(label="Step-by-step Output", lines=16)
            btn_fp.click(run_forward_propagation, inputs=[x1_fp, x2_fp], outputs=out_fp)

        # ---------------- Tab 3: Backpropagation ----------------
        with gr.Tab("3. Backpropagation"):
            gr.Markdown("### Manual backpropagation (NumPy) learning XOR")
            with gr.Row():
                lr_bp = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Learning Rate")
                epochs_bp = gr.Slider(100, 20000, value=10000, step=100, label="Epochs")
            btn_bp = gr.Button("Train", variant="primary")
            with gr.Row():
                log_bp = gr.Textbox(label="Training Log", lines=18)
                plot_bp = gr.Plot(label="Loss Curve")
            btn_bp.click(run_backpropagation, inputs=[lr_bp, epochs_bp], outputs=[log_bp, plot_bp])

        # ---------------- Tab 4: Activation Functions ----------------
        with gr.Tab("4. Activation Functions"):
            gr.Markdown("### Sigmoid, Tanh, ReLU, Leaky ReLU + Softmax")
            softmax_in = gr.Textbox(value="1.0, 2.0, 3.0", label="Softmax input (comma-separated)")
            btn_af = gr.Button("Generate", variant="primary")
            plot_af = gr.Plot(label="Activation Functions")
            out_af = gr.Textbox(label="Softmax Result", lines=4)
            btn_af.click(run_activation_functions, inputs=softmax_in, outputs=[plot_af, out_af])
            demo.load(run_activation_functions, inputs=softmax_in, outputs=[plot_af, out_af])

        # ---------------- Tab 5: Loss Functions ----------------
        with gr.Tab("5. Loss Functions"):
            gr.Markdown("### MSE, Binary Cross-Entropy, Categorical Cross-Entropy")
            loss_type = gr.Dropdown(
                choices=["MSE", "Binary Cross-Entropy", "Categorical Cross-Entropy"],
                value="MSE", label="Loss Type",
            )
            with gr.Row():
                y_true_in = gr.Textbox(value=LOSS_DEFAULTS["MSE"][0], label="y_true (comma-separated)")
                y_pred_in = gr.Textbox(value=LOSS_DEFAULTS["MSE"][1], label="y_pred (comma-separated)")
            btn_lf = gr.Button("Compute Loss", variant="primary")
            out_lf = gr.Textbox(label="Result", lines=8)
            loss_type.change(loss_type_change, inputs=loss_type, outputs=[y_true_in, y_pred_in])
            btn_lf.click(run_loss_function, inputs=[loss_type, y_true_in, y_pred_in], outputs=out_lf)

        # ---------------- Tab 6: Gradient Descent ----------------
        with gr.Tab("6. Gradient Descent"):
            gr.Markdown("### Linear regression fit via gradient descent")
            with gr.Row():
                x_in = gr.Textbox(value="1, 2, 3, 4, 5", label="X (comma-separated)")
                y_in = gr.Textbox(value="3, 5, 7, 9, 11", label="y (comma-separated)")
            with gr.Row():
                lr_gd = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Learning Rate")
                epochs_gd = gr.Slider(10, 5000, value=1000, step=10, label="Epochs")
            btn_gd = gr.Button("Train", variant="primary")
            log_gd = gr.Textbox(label="Training Log", lines=14)
            plot_gd = gr.Plot(label="Loss Curve & Fitted Line")
            btn_gd.click(run_gradient_descent, inputs=[x_in, y_in, lr_gd, epochs_gd], outputs=[log_gd, plot_gd])

        # ---------------- Tab 7: Neural Network (PyTorch XOR) ----------------
        with gr.Tab("7. Neural Network (PyTorch)"):
            gr.Markdown("### A small PyTorch neural network learning XOR")
            with gr.Row():
                lr_nn = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Learning Rate")
                epochs_nn = gr.Slider(100, 20000, value=10000, step=100, label="Epochs")
            btn_nn = gr.Button("Train", variant="primary")
            with gr.Row():
                log_nn = gr.Textbox(label="Training Log", lines=18)
                plot_nn = gr.Plot(label="Loss Curve")
            btn_nn.click(run_neural_network, inputs=[lr_nn, epochs_nn], outputs=[log_nn, plot_nn])

        # ---------------- Tab 8: Optimizer Comparison ----------------
        with gr.Tab("8. Optimizer Comparison"):
            gr.Markdown("### SGD vs SGD+Momentum vs Adam on the XOR problem")
            epochs_oc = gr.Slider(100, 5000, value=2000, step=100, label="Epochs")
            btn_oc = gr.Button("Run Comparison", variant="primary")
            plot_oc = gr.Plot(label="Loss Curves")
            out_oc = gr.Textbox(label="Final Losses", lines=5)
            btn_oc.click(run_optimizer_comparison, inputs=epochs_oc, outputs=[plot_oc, out_oc])

        # ---------------- Tab 9: Data Augmentation ----------------
        with gr.Tab("9. Data Augmentation"):
            gr.Markdown("### torchvision augmentations (upload an image, or leave empty to use the sample cat photo)")
            img_da = gr.Image(type="pil", label="Image (optional)")
            btn_da = gr.Button("Apply Augmentations", variant="primary")
            gallery_da = gr.Gallery(label="Augmented Versions", columns=4, height="auto")
            btn_da.click(run_data_augmentation, inputs=img_da, outputs=gallery_da)

        # ---------------- Tab 10: Transfer Learning ----------------
        with gr.Tab("10. Transfer Learning"):
            gr.Markdown(
                "### ResNet18 transfer learning (frozen backbone) on a tiny cat/dog dataset\n"
                "Only the final layer is trained; the pretrained ImageNet backbone is frozen. "
                "First run needs internet access to download ImageNet weights."
            )
            epochs_tl = gr.Slider(1, 15, value=5, step=1, label="Epochs")
            btn_tl = gr.Button("Run Transfer Learning", variant="primary")
            log_tl = gr.Textbox(label="Training Log", lines=18)
            btn_tl.click(run_transfer_learning, inputs=epochs_tl, outputs=log_tl)

        # ---------------- Tab 11: Mini Project ----------------
        with gr.Tab("Mini Project: Transfer Learning + Fine-Tuning"):
            gr.Markdown(
                "### Mini Project - Two-phase training on the cat/dog dataset\n"
                "**Phase 1:** train only the final layer (frozen backbone).\n"
                "**Phase 2:** unfreeze `layer4` and fine-tune with a smaller learning rate.\n\n"
                "After training, try the model on your own image below."
            )
            with gr.Row():
                epochs_transfer = gr.Slider(1, 15, value=5, step=1, label="Phase 1 Epochs (Transfer Learning)")
                epochs_finetune = gr.Slider(1, 15, value=5, step=1, label="Phase 2 Epochs (Fine-Tuning)")
            btn_mp = gr.Button("Run Full Pipeline", variant="primary")
            log_mp = gr.Textbox(label="Training Log", lines=22)

            gr.Markdown("---\n#### Predict on your own image")
            with gr.Row():
                img_predict = gr.Image(type="pil", label="Upload a cat or dog photo")
                out_predict = gr.Textbox(label="Prediction", lines=6)
            btn_predict = gr.Button("Predict", interactive=False)

            btn_mp.click(run_mini_project, inputs=[epochs_transfer, epochs_finetune],
                         outputs=[log_mp, btn_predict])
            btn_predict.click(predict_mini_project, inputs=img_predict, outputs=out_predict)

    gr.Markdown(
        "---\n*DAY 15 - Neural Networks & Deep Learning: perceptron, forward/back "
        "propagation, activation & loss functions, gradient descent, a PyTorch "
        "neural net, optimizer comparison, data augmentation, and transfer "
        "learning / fine-tuning.*"
    )


if __name__ == "__main__":
    demo.launch()