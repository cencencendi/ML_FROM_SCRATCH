import numpy as np
from ._base import LinearRegression


class Lasso(LinearRegression):
    def __init__(self, alpha: float, max_iter: int, tol: float) -> None:
        """
        Initialize the Lasso Regression model.

        Parameters
        ----------
        alpha : float
            Regularization strength.
        max_iter : int
            Maximum number of iterations for optimization.
        tol : float
            Tolerance to declare convergence.

        Returns
        -------
        None
        """
        # Set regularization strength, maximum iterations, and tolerance
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """
        Fit the Lasso regression model to the input data using coordinate descent.

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

        # Copy input data and target values
        self._X = X.copy()
        self._y = y.copy()

        # Get the number of samples and features in the input data
        n_samples, n_features = self._X.shape

        # Initialize design matrix by adding a column of ones
        self._X = np.concatenate([self._X, np.ones(shape=(n_samples, 1))], axis=1)

        # Initialize coefficients (theta) with zeros
        self.theta = np.zeros(n_features + 1)

        # Iterative optimization using coordinate descent
        for _ in range(self.max_iter):
            for j in range(n_features + 1):
                # Calculate partial derivative with respect to theta[j]
                rho_j = self._X[:, j] @ (y - self._X @ self.theta + self._X[:, j] * self.theta[j])

                # Calculate squared L2 norm of the j-th column of X
                z_j = self._X[:, j] @ self._X[:, j]

                # Update theta[j] using soft thresholding
                if j == n_features:  # Intercept
                    self.theta[j] = rho_j
                else:
                    self.theta[j] = self.soft_threshold(rho_j) / z_j

            # Check for convergence or reaching the maximum iterations
            if self.compute_rss() < self.tol:
                break

        # Store the final coefficients
        self.coeff_ = self.theta

    def soft_threshold(self, rho):
        """
        Apply the soft-thresholding function element-wise.

        Parameters
        ----------
        rho : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Result of the soft-thresholding function applied element-wise.
        """
        return np.sign(rho) * np.maximum(0, np.abs(rho) - self.alpha)

    def compute_rss(self):
        """
        Calculate the Residual Sum of Squares (RSS) for the Lasso Regression model.
        cost = (1 / (2 * n_samples)) * ||y-Xw||^2_2 + alpha * sum ||w||_1

        Returns
        -------
        float
            The computed RSS value.
        """
        _, n_features = self._X.shape
        y_pred = np.dot(self._X, self.theta)
        return (1 / 2) * np.mean(np.dot(self._y - y_pred, self._y - y_pred)) + self.alpha * np.sum(
            abs(self.theta[: n_features - 1])
        )
