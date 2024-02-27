import numpy as np
import copy

from ..model_selection import KFold
from ..metrics import __all__


def cross_val_score(estimator, X, y, n_cv=5, scoring="mean_squared_error"):
    """
    Perform cross-validated scoring for an estimator.

    Parameters
    ----------
    estimator : object
        An instance of the estimator to be evaluated.
    X : array-like or pd.DataFrame
        Input features.
    y : array-like or pd.Series
        Target values.
    n_cv : int, optional
        Number of cross-validation folds. Default is 5.
    scoring : str, optional
        The scoring metric to use. Default is "mean_squared_error".

    Returns
    -------
    tuple
        Tuple containing lists of training scores and testing scores for each fold.
    """
    # Make copies of input data to avoid modifying the original data
    X = np.array(X).copy()
    y = np.array(y).copy()

    # Create KFold object for cross-validation
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)

    # Get the scoring function based on the provided metric
    scoring = __all__[scoring]

    # Lists to store training and testing scores for each fold
    score_train_list = []
    score_test_list = []

    # Perform cross-validation
    for train_ids, test_ids in kf.split(X):
        X_train, X_test = X[train_ids], X[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]

        # Create a deep copy of the estimator to avoid modifying the original model
        model = copy.deepcopy(estimator)

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the training and testing sets
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate the scores using the specified metric
        score_train = scoring(y_true=y_train, y_pred=y_pred_train)
        score_test = scoring(y_true=y_test, y_pred=y_pred_test)

        # Append scores to the lists
        score_train_list.append(score_train)
        score_test_list.append(score_test)

    # Return the lists of training and testing scores for each fold
    return score_train_list, score_test_list
