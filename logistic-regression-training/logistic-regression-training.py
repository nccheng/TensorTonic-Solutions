import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1, 1)

    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0

    for _ in range(steps):
        preds = _sigmoid(X @ w + b)
        dw = (1 / m) * X.T @ (preds - y)
        db = np.mean(preds - y)

        w = w - lr * dw
        b = b - lr * db

    return (w.flatten(), b)