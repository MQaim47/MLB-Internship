from sklearn.model_selection import train_test_split
import numpy as np

# Sample Dataset
X = np.arange(100).reshape(50, 2)
y = np.arange(50)

# Train = 70%
# Temp = 30%

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

# Validation = 15%
# Test = 15%

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))