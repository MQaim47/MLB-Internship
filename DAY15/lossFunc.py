import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def binary_cross_entropy(y_true, y_pred):
    
    epsilon = 1e-15
    
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss

def categorical_cross_entropy(y_true, y_pred):

    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred))

    return loss


y_true_mse = np.array([1, 0, 1])
y_pred_mse = np.array([0.8, 0.2, 0.9])

mse_loss = mse(y_true_mse, y_pred_mse)

y_true_bce = np.array([1, 0, 1, 0])

y_pred_bce = np.array([0.9, 0.2, 0.8, 0.1])

bce_loss = binary_cross_entropy(
    y_true_bce,
    y_pred_bce
)


y_true_cce = np.array([1, 0, 0])

y_pred_cce = np.array([0.8, 0.1, 0.1])

cce_loss = categorical_cross_entropy(
    y_true_cce,
    y_pred_cce
)


print("MSE Loss :", mse_loss)

print("Binary Cross Entropy :", bce_loss)

print("Categorical Cross Entropy :", cce_loss)