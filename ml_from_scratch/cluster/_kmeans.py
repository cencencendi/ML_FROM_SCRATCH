import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,  # Number of clusters to form
        init: str = "kmeans++",  # Initialization method for centroids
        max_iter: int = 300,  # Maximum number of iterations
        tol: float = 1e-4,  # Tolerance to declare convergence
        random_state: int = 42,  # Random seed for reproducibility
    ) -> None:
        """
        Initialize KMeans clustering algorithm.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters to form, by default 8.
        init : str, optional
            Initialization method for centroids ('kmeans++' or 'random'), by default "kmeans++".
        max_iter : int, optional
            Maximum number of iterations, by default 300.
        tol : float, optional
            Tolerance to declare convergence, by default 1e-4.
        random_state : int, optional
            Random seed for reproducibility, by default 42.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> None:
        """
        Fit KMeans clustering to the given data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        """
        # Set the random seed for reproducibility
        np.random.seed(self.random_state)

        # Make a copy of the input data
        self._X = np.array(X).copy()

        # Get the number of samples and features
        self.n_samples, self.n_features = self._X.shape

        # Initialize centroids using the chosen method
        self.centroids = self._init_centroid()

        # Iterate until convergence or max iterations
        for _ in range(self.max_iter):
            # Compute distances and assign clusters
            distance, self.labels = self._calculate_nearest_distance(
                x=self._X, centroids=self.centroids
            )

            # Vectorized centroid update
            new_centroids = np.array(
                [self._X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)]
            )

            # inertia_: Sum of squared distances of samples to their closest cluster center
            self.inertia_ = np.sum(distance**2)

            # Check for convergence
            if self.inertia_ < self.tol:
                break

            # Update centroids
            self.centroids = new_centroids

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute the Euclidean distance between each target point and all data points.

        Parameters
        ----------
        x : np.ndarray
            Target points.
        centroids : np.ndarray
            Centroids of each cluster.

        Returns
        -------
        np.ndarray
            Euclidean distances.
        """
        # Compute Euclidean distances
        return np.sqrt(((x - centroids) ** 2).sum(axis=2))

    def _init_centroid(self):
        """
        Initialize centroids based on the chosen method.

        Returns
        -------
        np.ndarray
            Initialized centroids.
        """
        # Set the random seed for reproducibility
        np.random.seed(self.random_state)

        # Initialize centroids using k-means++ algorithm
        if self.init == "kmeans++":

            # Initialize empty centroids with shape (k, n_features)
            centroids = np.empty(shape=(self.n_clusters, self.n_features), dtype=float)

            # Randomly choose the first centroid
            centroids[0] = self._X[np.random.randint(0, self.n_samples)]

            for i in range(1, self.n_clusters):
                # Calculate distance to the nearest centroid for each point
                nearest_distance = self._calculate_nearest_distance(
                    x=self._X, centroids=centroids[:i], return_label=False
                )
                # Calculate probabilities for selecting the next centroid
                probabilities = nearest_distance**2 / np.sum(nearest_distance**2)
                # Choose the next centroid with probabilities proportional to distance
                centroids[i] = self._X[np.random.choice(a=self.n_samples, p=probabilities)]

            return centroids

        # Initialize centroids randomly within data range
        if self.init == "random":

            # Calculate min and max values for each feature
            min_, max_ = self._X.min(axis=0), self._X.max(axis=0)
            return np.random.rand(self.n_clusters, self.n_features) * (max_ - min_) + min_

    def predict(self, X: np.ndarray):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Index of the cluster each sample belongs to.
        """
        # Compute distances and assign clusters for each data point
        return self._calculate_nearest_distance(x=X, centroids=self.centroids)[1]

    def _calculate_nearest_distance(
        self, x: np.ndarray, centroids: np.ndarray, return_label: bool = True
    ):
        """
        Calculate the nearest distance from each data point to the centroids.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        centroids : np.ndarray
            Centroids of the clusters.
        return_label : bool, optional
            Whether to return labels along with distances (default is True).

        Returns
        -------
        np.ndarray
            Array of nearest distances from each data point to the centroids.
        np.ndarray (optional)
            Index of the centroid each sample is closest to (if return_label is True).
        """
        # Compute distances between data points and centroids
        distances = self._compute_distance(x=x[:, np.newaxis], centroids=centroids)

        # Find the nearest distance for each data point
        nearest_distance = np.min(distances, axis=1)

        if return_label:
            # Return labels along with distances
            return nearest_distance, np.argmin(distances, axis=1)

        # Return only the nearest distances
        return nearest_distance
