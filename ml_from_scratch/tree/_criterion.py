import numpy as np


def Gini(y):
    """
    Calculate Gini impurity for a set of labels.

    Parameters
    -----------
    y : np.ndarray
        Labels.

    Returns
    -------
    float
        Gini impurity value.
    """
    _, unique_counts = np.unique(y, return_counts=True)
    N_m = len(y)

    node_impurity = 0
    for counts in unique_counts:
        p_m_k = counts / N_m
        node_impurity += p_m_k * (1 - p_m_k)

    return node_impurity


def Entropy(y):
    """
    Calculate entropy for a set of labels.

    Parameters
    ----------
    y : np.ndarray
        Labels.

    Returns
    -------
    float
        Entropy value.
    """
    _, unique_counts = np.unique(y, return_counts=True)
    N_m = len(y)

    node_impurity = 0
    for counts in unique_counts:
        p_m_k = counts / N_m
        node_impurity += p_m_k * np.log(p_m_k)

    return -node_impurity


def LogLoss(y):
    """
    Compute the Logarithmic Loss (LogLoss) for binary classification.

    Parameters
    ----------
    y : np.ndarray
        True labels (binary, 0 or 1).

    Returns
    -------
    float
        Logarithmic Loss value.
    """
    epsilon = 1e-15  # Small value to avoid log(0)
    y = np.clip(y, epsilon, 1 - epsilon)  # Clip probabilities to avoid log(0) or log(1)

    log_loss = -np.mean(y * np.log(y) + (1 - y) * np.log(1 - y))
    return log_loss


def MSE(y):
    """
    Calculate Mean Squared Error (MSE) for a set of values.

    Parameters
    ----------
    y : np.ndarray
        Values.

    Returns
    -------
    float
        Mean Squared Error value.
    """
    c_m = np.mean(y)
    return np.mean((y - c_m) ** 2)


def MAE(y):
    """
    Calculate Mean Absolute Error (MAE) for a set of values.

    Parameters
    ----------
    y : np.ndarray
        Values.

    Returns
    -------
    float
        Mean Absolute Error value.
    """
    c_m = np.median(y)
    return np.mean(np.abs(y - c_m))
