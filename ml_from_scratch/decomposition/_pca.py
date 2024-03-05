import numpy as np


class PCA:
    def __init__(self, n_components: int = None) -> None:
        """
        Initialize the PCA model with the option to specify the number of principal components.

        Parameters
        ----------
        n_components : int, optional
            The number of principal components to compute. If not specified, all components are used.

        Returns
        -------
        None
        """
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model to the input data by computing the principal components.

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

        # Set the number of components to the number of features if not specified
        if self.n_components is None:
            self.n_components = n_features

        # Calculate the mean of each feature
        self.mean_ = np.mean(self._X, axis=0)

        # Center the data by subtracting the mean
        X_centered = self._X - self.mean_

        # Calculate the covariance matrix of the centered data
        cov_mat = np.cov(X_centered, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eig_val, eig_vec = np.linalg.eig(cov_mat)

        # Sort the eigenvalues and eigenvectors in descending order of the eigenvalues
        sorted_ids = (-1 * eig_val).argsort()
        sorted_eig_val = eig_val[sorted_ids]
        sorted_eig_vec = eig_vec[:, sorted_ids]

        # Calculate the explained variance ratio
        explained_variance_ratio = sorted_eig_val / np.sum(sorted_eig_val)

        # Store the top n_components eigenvectors, eigenvalues, and explained variance ratio
        self.components_ = sorted_eig_vec[:, : self.n_components].T
        self.explained_variance_ = sorted_eig_val[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]

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
        # Make a copy of the input data
        X = np.array(X).copy()

        # Center the data by subtracting the mean
        if self.mean_ is not None:
            X = X - self.mean_

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
        return np.dot(X, self.components_) + self.mean_
