import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy score for a classification model.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_pred : np.ndarray
        Predicted class labels.

    Returns
    -------
    float
        The accuracy score.
    """
    # Calculate the accuracy by comparing true and predicted class labels

    return np.mean(y_true == y_pred)
