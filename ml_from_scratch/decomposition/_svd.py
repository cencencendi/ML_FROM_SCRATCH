import numpy as np


def _generate_unit_vector(n: int):
    """
    Generate a random unit vector of dimension n.

    Parameters
    ----------
    n : int
        The dimension of the unit vector.

    Returns
    -------
    np.ndarray
        A unit vector with n dimensions.
    """
    mean = 0.0  # Mean of the normal distribution to generate components
    std = 1.0  # Standard deviation of the normal distribution

    # Generate a random vector from a normal distribution and normalize it
    vec = np.random.normal(loc=mean, scale=std, size=n)
    return vec / np.linalg.norm(vec)


class SVD:
    def __init__(self, n_components: int = None, tol: float = 1e-10) -> None:
        """
        Initialize the SVD model with the option to specify the number of singular values and a tolerance for convergence.

        Parameters
        ----------
        n_components : int, optional
            The number of singular values and vectors to compute. If not specified, all are used.
        tol : float, optional
            The convergence tolerance for power iteration. Default is 1e-10.

        Returns
        -------
        None
        """
        self.n_components = n_components
        self.tol = tol

    def fit(self, X: np.ndarray):
        """
        Fit the SVD model to the input data by computing its singular value decomposition.

        Parameters
        ----------
        X : np.ndarray
            The input data, where rows are samples and columns are features.

        Returns
        -------
        None
        """
        # Make a copy of the input data to ensure the original data is not modified
        self._X = np.array(X).copy()

        # Determine the shape of the input data
        _, n_features = self._X.shape

        # If the number of components is not specified, use all features
        if self.n_components is None:
            self.n_components = n_features

        change_of_basis = []

        for i in range(n_features):
            A = self._X.copy()

            # Remove the contribution of all previously found singular vectors
            for u, sigma, v in change_of_basis[:i]:
                A -= sigma * np.outer(u, v)

            # Perform power iteration on the adjusted matrix to find the next singular vector and value
            u, sigma, v = self.power_iterate(A)

            # Store the singular vectors and value
            change_of_basis.append((u, sigma, v))

        # Unpack and store the singular vectors and values
        u_vec, sigma_vec, v_vec_T = [np.array(x) for x in zip(*change_of_basis)]

        # Calculate and store the explained variance and its ratio
        explained_variance = sigma_vec**2
        explained_variance_ratio = explained_variance / np.sum(explained_variance)

        self.components_ = v_vec_T[:, : self.n_components].T
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]

    def power_iterate(self, A):
        """
        Perform the power iteration method to find the largest singular value and corresponding singular vectors of A.

        Parameters
        ----------
        A : np.ndarray
            The matrix to decompose.

        Returns
        -------
        tuple
            The left singular vector, singular value, and right singular vector.
        """
        _, n_features = A.shape

        # Initialize with a random unit vector
        x_0 = _generate_unit_vector(n=n_features)
        x_curr = x_0.copy()
        err = np.inf

        # Iterate until the vector converges
        while err > self.tol:
            x_prev = x_curr.copy()
            x_curr = A.T @ A @ x_prev
            x_curr /= np.linalg.norm(x_curr)

            # Check for convergence
            err = np.linalg.norm(x_curr - x_prev)
            if err < self.tol:
                break

        # Normalize and calculate singular vectors and value
        v = x_curr
        sigma = np.linalg.norm(A @ v)
        u = (A @ v) / sigma

        return u, sigma, v

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray
            The input data to transform, where rows are samples and columns are features.

        Returns
        -------
        np.ndarray
            The data projected into the principal component space.
        """
        # Project the data onto the principal component axes
        return np.dot(X, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model to X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit and transform.

        Returns
        -------
        np.ndarray
            The data projected into the principal component space.
        """
        # Fit the PCA model to the data
        self.fit(X)

        # Transform the data using the fitted model
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to its original space.

        Parameters
        ----------
        X : np.ndarray
            The data in the principal component space to invert back to the original space.

        Returns
        -------
        np.ndarray
            The data in the original feature space.
        """
        # Project the data back to the original space
        return np.dot(X, self.components_)
