import numpy as np


class SVC:
    def __init__(self, c: float = 1.0, tol: float = 1e-5, max_passes: int = 2) -> None:
        """
        Initialize the Support Vector Classifier.

        Parameters
        ----------
        c : float, optional
            Regularization parameter (default is 1.0).
        tol : float, optional
            Tolerance for convergence (default is 1e-5).
        max_passes : int, optional
            Maximum number of passes without alpha changes (default is 1).

        Returns
        -------
        None
        """
        self.c = c
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Support Vector Classifier to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.

        Returns
        -------
        None
        """
        # Copy the input arrays to avoid modifying the original data.
        self._X = np.copy(X)
        self._y = np.copy(y)

        # Get the number of samples and features from the training data.
        n_samples, n_features = self._X.shape

        # Initialize alpha (lagrange multipliers) and b (bias).
        self.alpha = np.zeros(n_samples, dtype=float)
        self.b = 0

        # Initialize the number of passes for the training loop.
        passes = 0

        # Start the training loop, continuing until max_passes is reached.
        while passes <= self.max_passes:
            # Initialize the counter for the number of changed alphas in the loop.
            num_changed_alphas = 0

            # Iterate over each sample in the training data.
            for i in range(n_samples):
                # Compute the error for the current sample.
                E_i = np.dot(self.alpha * self._y, (self._X[i] @ self._X.T)) + self.b - self._y[i]

                # Check if the current alpha violates the KKT conditions.
                if (self._y[i] * E_i < -self.tol and self.alpha[i] < self.c) or (
                    self._y[i] * E_i > self.tol and self.alpha[i] > 0
                ):
                    # Choose a random j different from i.
                    j = np.random.choice(np.setdiff1d(np.arange(n_samples), i))

                    # Compute the error for the randomly chosen sample.
                    E_j = (
                        np.dot(self.alpha * self._y, (self._X[j] @ self._X.T)) + self.b - self._y[j]
                    )

                    # Save the old values of alpha_i and alpha_j.
                    a_i_old = self.alpha[i]
                    a_j_old = self.alpha[j]

                    # Determine the bounds L and H for alpha_j.
                    if y[i] != y[j]:
                        L = np.maximum(0.0, self.alpha[j] - self.alpha[i])
                        H = np.minimum(self.c, self.c + self.alpha[j] - self.alpha[i])
                    else:
                        L = np.maximum(0.0, self.alpha[i] + self.alpha[j] - self.c)
                        H = np.minimum(self.c, self.alpha[i] + self.alpha[j])

                    # Check if L is equal to H, if so, continue to the next iteration.
                    if L == H:
                        continue

                    # Compute the parameter eta for updating alpha_j.
                    eta = (
                        2 * np.dot(self._X[i], self._X[j])
                        - self._X[i] @ self._X[i]
                        - self._X[j] @ self._X[j]
                    )

                    # If eta is non-positive, continue to the next iteration.
                    if eta >= 0:
                        continue

                    # Update alpha_j based on the computed values.
                    a_j = self.alpha[j] - (self._y[j] * (E_i - E_j) / eta)
                    self.alpha[j] = H if a_j > H else a_j if L < a_j < H else L

                    # Check if alpha_j has changed significantly.
                    if np.abs(self.alpha[j] - a_j_old) < 1e-5:
                        continue

                    # Update alpha_i based on the change in alpha_j.
                    self.alpha[i] = self.alpha[i] + self._y[i] * self._y[j] * (
                        a_j_old - self.alpha[j]
                    )

                    # Update bias terms b1 and b2.
                    b1 = (
                        self.b
                        - E_i
                        - self._y[i] * (self.alpha[i] - a_i_old) * (self._X[i] @ self._X[i])
                        - self._y[j] * (self.alpha[j] - a_j_old) * (self._X[i] @ self._X[j])
                    )

                    b2 = (
                        self.b
                        - E_j
                        - self._y[i] * (self.alpha[i] - a_i_old) * (self._X[i] @ self._X[j])
                        - self._y[j] * (self.alpha[j] - a_j_old) * (self._X[j] @ self._X[j])
                    )

                    # Update the bias term based on the conditions.
                    self.b = (
                        b1
                        if 0 < self.alpha[i] < self.c
                        else b2
                        if 0 < self.alpha[j] < self.c
                        else (b1 + b2) / 2
                    )

                    # Increment the counter for changed alphas.
                    num_changed_alphas += 1

            # Check if no alphas have changed, increment passes, otherwise reset passes.
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # Compute the coefficients of the support vector machine.
        self.coeff_ = np.dot(self.alpha * self._y, self._X)
        self.intercept_ = self.b

    def predict(self, X: np.ndarray):
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        # Calculate the raw predictions using the dot product of input features and coefficients, plus bias.
        pred = np.dot(X, self.coeff_) + self.b

        # Apply a threshold to the raw predictions to determine the class labels (1 or -1).
        return np.where(pred > 0, 1, -1)
