from ._base import BaseEnsemble
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
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


class RandomForestRegressor(BaseEnsemble):
    def __init__(
        self,
        n_estimators: int = 100,
        max_features="sqrt",
        random_state: int = 42,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
    ) -> None:
        """
        Initializes the RandomForestRegressor instance.

        Parameters
        ----------
        n_estimators : int, optional
            The number of regressors (copies) in the ensemble.
        max_features : int, str, or None, optional
            The maximum number of features to consider for each regressor.
            If None, all features will be used. If int or str, it will be applied uniformly across regressors.
        random_state : int, optional
            Seed for reproducibility.
        criterion : str, optional
            The function to measure the quality of a split. Supported criteria are "mae" for the mean absolute error
            and "mse" for the mean squared error.
        max_depth : int, optional
            The maximum depth of the tree. If None, the tree is expanded until all leaves contain less than
            min_samples_split samples.
        min_samples_split : int, optional
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node.
        min_impurity_decrease : float, optional
            A node will be split if this split induces a decrease of the impurity greater than or equal to
            this value.

        Returns
        -------
        None
        """
        self.agg_func = _get_average_vote
        estimator = DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        super().__init__(estimator, n_estimators, max_features, random_state)


class RandomForestClassifier(BaseEnsemble):
    def __init__(
        self,
        n_estimators: int = 100,
        max_features="sqrt",
        random_state: int = 42,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
    ) -> None:
        """
        Initializes the RandomForestClassifier instance.

        Parameters
        ----------
        n_estimators : int, optional
            The number of classifiers (copies) in the ensemble.
        max_features : int, str, or None, optional
            The maximum number of features to consider for each classifier.
            If None, all features will be used. If int or str, it will be applied uniformly across classifiers.
        random_state : int, optional
            Seed for reproducibility.
        criterion : str, optional
            The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        max_depth : int, optional
            The maximum depth of the tree. If None, the tree is expanded until all leaves contain less than
            min_samples_split samples.
        min_samples_split : int, optional
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node.
        min_impurity_decrease : float, optional
            A node will be split if this split induces a decrease of the impurity greater than or equal to
            this value.

        Returns
        -------
        None
        """
        self.agg_func = _get_majority_vote
        estimator = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        super().__init__(estimator, n_estimators, max_features, random_state)
