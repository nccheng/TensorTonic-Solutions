import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    m, n = X.shape          # Fix 1
    w = np.zeros((n, 1))
    b = 0.0

    for _ in range(steps):
        preds = _sigmoid(X @ w + b)
        dw = (1 / m) * X.T @ (preds - y)   # Fix 2 & 3
        db = np.mean(preds - y)
        w -= lr * dw
        b -= lr * float(db)                 # Fix 3

    return (w.flatten(), b)
    