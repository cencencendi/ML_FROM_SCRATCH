import numpy as np


class NearestNeighbors:
    def __init__(self, n_neighbors: int = 5, p: int = 2) -> None:
        """
        Initialize the NearestNeighbors class.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors to consider (default is 5).
        p : int, optional
            The power parameter for the Minkowski distance (default is 2 for Euclidean distance).

        Returns
        -------
        None
        """
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model with training data.

        Parameters
        ----------
        X : np.ndarray
            Training data, a 2D numpy array.
        y : np.ndarray
            Target values, a 1D numpy array.

        Returns
        -------
        None
        """
        # Store a copy of the training data
        self._X = np.copy(X)
        self._y = np.copy(y)

    def _compute_distance(self, x_target: np.ndarray) -> np.ndarray:
        """
        Compute the Minkowski distance between each target point and all data points.

        Parameters
        ----------
        x_target : np.ndarray
            Target points.

        Returns
        -------
        np.ndarray
            Minkowski distances.
        """
        # Compute Minkowski distances using the specified power parameter
        return np.sum(np.abs(x_target - self._X) ** self.p, axis=2) ** (1.0 / self.p)

    def _kneighbors(self, x_target: np.ndarray, return_distance: bool = True):
        """
        Find the k-nearest neighbors for each target point.

        Parameters
        ----------
        x_target : np.ndarray
            Target points for which neighbors are to be found.
        return_distance : bool, optional
            If True, also return the distances to the neighbors.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Indices of k-nearest neighbors for each target point.
            If return_distance is True, also return the distances.
        """
        # Expand dimensions to match the shape for computation
        x_target = x_target[:, np.newaxis, :]

        # Compute Minkowski distances for each target point
        distances = self._compute_distance(x_target=x_target)

        if return_distance:
            # Return both indices and distances if specified
            return (
                np.argsort(distances)[:, : self.n_neighbors],
                np.sort(distances)[:, : self.n_neighbors],
            )

        # Return only indices if distances are not needed
        return np.argsort(distances)[:, : self.n_neighbors]
