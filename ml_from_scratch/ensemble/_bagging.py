from ._base import BaseEnsemble
import numpy as np
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from collections import Counter


# Function to calculate the average vote from an array of numerical values
def _get_average_vote(y):
    return np.mean(y, axis=0)


# Function to calculate the majority vote from an array of labels
def _get_majority_vote(y):
    n_samples = y.shape[1]
    y_pred = np.empty(shape=(n_samples,))
    for i in range(n_samples):
        y_pred[i] = Counter(y[:, i]).most_common(1)[0][0]

    return y_pred


class BaggingRegressor(BaseEnsemble):
    def __init__(
        self, estimator=None, n_estimators: int = 10, max_features=None, random_state: int = 42
    ) -> None:
        """
        Initializes the BaggingRegressor instance.

        Parameters
        ----------
        estimator : object, optional
            The base regressor to be used in the ensemble. If None, a DecisionTreeRegressor is used.
        n_estimators : int, optional
            The number of regressors (copies) in the ensemble.
        max_features : int, str, or None, optional
            The maximum number of features to consider for each regressor.
            If None, all features will be used. If int or str, it will be applied uniformly across regressors.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        None
        """
        self.agg_func = _get_average_vote
        if estimator is None:
            estimator = DecisionTreeRegressor()
        super().__init__(estimator, n_estimators, max_features, random_state)


class BaggingClassifier(BaseEnsemble):
    def __init__(
        self, estimator=None, n_estimators: int = 10, max_features=None, random_state: int = 42
    ) -> None:
        """
        Initializes the BaggingClassifier instance.

        Parameters
        ----------
        estimator : object, optional
            The base classifier to be used in the ensemble. If None, a DecisionTreeClassifier is used.
        n_estimators : int, optional
            The number of classifiers (copies) in the ensemble.
        max_features : int, str, or None, optional
            The maximum number of features to consider for each classifier.
            If None, all features will be used. If int or str, it will be applied uniformly across classifiers.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        None
        """
        self.agg_func = _get_majority_vote
        if estimator is None:
            estimator = DecisionTreeClassifier()
        super().__init__(estimator, n_estimators, max_features, random_state)
