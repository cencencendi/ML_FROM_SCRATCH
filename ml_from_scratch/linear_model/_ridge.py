import numpy as np
from ._base import LinearRegression


class Ridge(LinearRegression):
    def __init__(self, alpha: float) -> None:
        """
        Initialize the RidgeRegression model.

        Parameters
        ----------
        alpha : float
                Regularization strength.

        Returns
        -------
        None
        """
        # Set regularization strength
        self.alpha = alpha

    def fit(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """
        Fit the ridge regression model to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.

        Returns
        -------
        None
        """
        # Get the number of samples and features in the input data
        n_samples, n_features = X.shape

        # Add a column of ones to the input data to represent the intercept term
        ones = np.ones(shape=(n_samples, 1))
        X_arr = np.concatenate([X, ones], axis=1)
        I = np.eye(n_features + 1, dtype=float)
        I[-1, -1] = 0

        # Compute the coefficients using the normal equation
        self.beta = np.linalg.inv(X_arr.T @ X_arr + self.alpha * I) @ np.dot(X_arr.T, y)

        # Separate the coefficients into coef_ and intercept_
        self.coef_, self.intercept_ = self.beta[:n_features], self.beta[-1]
