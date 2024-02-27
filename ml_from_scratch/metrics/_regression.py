import numpy as np


def mean_squarred_error(y_true: np.ndarray = None, y_pred: np.ndarray = None) -> float:
    """
    Calculate the mean squared error between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)
