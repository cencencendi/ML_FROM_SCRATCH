import numpy as np


class LinearRegression:
    def __init__(self) -> None:
        """
        Initialize the LinearRegression model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def fit(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """
        Fit the linear regression model to the input data.

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

        # Compute the coefficients using the normal equation
        self.theta = np.linalg.inv(X_arr.T @ X_arr) @ np.dot(X_arr.T, y)

        # Separate the coefficients into coef_ and intercept_
        self.coef_, self.intercept_ = self.theta[:n_features], self.theta[-1]

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """
        Make predictions using the learned coefficients.

        Parameters
        ----------
        X : np.ndarray
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        # Make predictions using the learned coefficients
        return X @ self.coef_ + self.intercept_
