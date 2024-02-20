import numpy as np


class LogisticRegression:
    def __init__(self, max_iter: int = 100, tol: float = 1e-4, learning_rate: float = 0.01) -> None:
        """
        Initialize the Logistic Regression model.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations for optimization (default is 100).
        tol : float, optional
            Tolerance to declare convergence (default is 1e-4).
        learning_rate : float, optional
            Learning rate for gradient descent (default is 0.01).

        Returns
        -------
        None
        """
        # Set the maximum number of iterations, convergence tolerance, and learning rate
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Logistic Regression model to the input data.

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
        self._X = np.copy(X)
        self._y = np.copy(y)

        # Get the number of features
        _, n_features = self._X.shape

        # Initialize coefficients and intercept to zeros
        self.coeff_, self.intercept_ = np.zeros(n_features), 0.0

        # Iterative optimization using gradient descent
        for _ in range(self.max_iter):
            # Make predictions using the current coefficients
            pred = self.predict_proba(self._X)

            # Compute gradients of coefficients and intercept
            grad_coeff = -np.mean(np.dot((self._y - pred), self._X))
            grad_intercept = -np.mean(self._y - pred)

            # Update coefficients and intercept using gradient descent
            self.coeff_ -= self.learning_rate * grad_coeff
            self.intercept_ -= self.learning_rate * grad_intercept

            # Check for convergence
            if np.all(grad_coeff) < self.tol and grad_intercept < self.tol:
                break

    def predict(self, X: np.ndarray):
        """
        Predict class labels for input data.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class labels (binary).
        """
        # Convert predicted probabilities to binary labels (0 or 1)
        return (self.predict_proba(X) > 0.5).astype("int")

    def predict_proba(self, X: np.ndarray):
        """
        Predict class probabilities for input data.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        # Apply the sigmoid function to the linear combination of input features and coefficients
        return self.sigmoid(np.dot(X, self.coeff_) + self.intercept_)

    def sigmoid(self, X: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Output values after applying the sigmoid function.
        """
        # Calculate the sigmoid function
        return 1 / (1 + np.exp(-X))
