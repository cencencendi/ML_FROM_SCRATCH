import numpy as np
from ._base import NearestNeighbors


class KNeighborsRegressor(NearestNeighbors):
    def __init__(self, n_neighbor: int = 5, p: int = 2, weights: str = "uniform") -> None:
        """
        Initialize the KNeighborsRegressor.

        Parameters
        ----------
        n_neighbor : int, optional
            Number of neighbors to consider (default is 5).
        p : int, optional
            The power parameter for the Minkowski distance (default is 2 for Euclidean distance).
        weights : str, optional
            Weight function used in prediction. Possible values are 'uniform' (default) or 'distance'.

        Returns
        -------
        None
        """
        # Initialize the NearestNeighbors base class with specified parameters
        super().__init__(n_neighbor, p)
        self.weights = weights

    def predict(self, x: np.ndarray) -> float:
        """
        Predict the target value for each input point.

        Parameters
        ----------
        x : np.ndarray
            Input data for prediction.

        Returns
        -------
        float
            Predicted target values.
        """
        # Get indices and distances of nearest neighbors
        nearest_neighbors_idx, nearest_distances = self._kneighbors(x_target=x)

        # Get weights for each neighbor
        weights_arr = self._get_weights(neighbors_distances=nearest_distances)

        if self.weights == "uniform":
            # Return the mean of target values for uniform weights
            return np.mean(self._y[nearest_neighbors_idx], axis=1)

        # Return the weighted mean of target values for distance-based weights
        return np.sum(self._y[nearest_neighbors_idx] * weights_arr, axis=1) / np.maximum(
            np.sum(weights_arr, axis=1), np.finfo(float).eps
        )
        # To avoid potential division by zero issues, add a small epsilon value to the denominator when calculating the weighted mean.

    def _get_weights(self, neighbors_distances: np.ndarray) -> np.ndarray:
        """
        Get the weights for each target point.

        Parameters
        ----------
        neighbors_distances : np.ndarray
            The K-closest neighbors distances.

        Returns
        -------
        np.ndarray
            Weights for each target point.
        """
        if self.weights == "uniform":
            return None

        # Handling potential division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            weights_arr = 1.0 / np.square(neighbors_distances)

        # Set weights_arr to 1.0 for points with zero distance
        weights_arr[neighbors_distances == 0] = 1.0

        return weights_arr
