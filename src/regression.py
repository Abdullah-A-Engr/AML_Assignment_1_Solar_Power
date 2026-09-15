import numpy as np

def hypothesis(X, theta):
    return X @ theta

def cost(X, y, theta):
    errors = hypothesis(X, theta) - y
    return np.mean(errors ** 2)

def fit_normal(X, y):
    # Equivalent to the assignment's (X^T X)^-1 X^T y.
    # pinv is used as a numerical safeguard if X^T X is ill-conditioned.
    XtX = X.T @ X
    try:
        theta = np.linalg.inv(XtX) @ X.T @ y
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(XtX) @ X.T @ y
    return theta

def fit_batch_gd(X, y, alpha, n_iters):
    theta = np.zeros(X.shape[1], dtype=float)
    history = []

    for _ in range(n_iters):
        pred = hypothesis(X, theta)
        error = y - pred
        theta = theta + (alpha / len(y)) * (X.T @ error)
        history.append(cost(X, y, theta))

    return theta, np.array(history)

def fit_sgd(X, y, alpha, n_epochs):
    theta = np.zeros(X.shape[1], dtype=float)
    history = []

    for _ in range(n_epochs):
        for i in range(len(y)):
            xi = X[i]
            error = y[i] - hypothesis(xi.reshape(1, -1), theta)[0]
            theta = theta + alpha * error * xi
        history.append(cost(X, y, theta))

    return theta, np.array(history)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def clip_predictions(y_pred):
    return np.maximum(y_pred, 0.0)
