import numpy as np
from ._base import NearestNeighbors


class KNeighborsClassifier(NearestNeighbors):
    def __init__(self, n_neighbors: int = 5, p: int = 2, weights: str = "uniform") -> None:
        """
        Initialize the KNeighborsClassifier.

        Parameters
        ----------
        n_neighbors : int, optional
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
        super().__init__(n_neighbors, p)
        self.weights = weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for each input point.

        Parameters
        ----------
        x : np.ndarray
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        return self._predict_proba(x_target=x)

    def _predict_proba(self, x_target: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each input point.

        Parameters
        ----------
        x_target : np.ndarray
            Target points for which class probabilities are to be calculated.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        # Get the indices and distances of nearest neighbors
        nearest_neighbors_idx, neighbors_distances = self._kneighbors(x_target=x_target)

        # Get weights for each neighbor
        weights_arr = self._get_weights(neighbors_distances=neighbors_distances)

        # Get class probabilities for each input point
        classes = set(self._y)
        majority_vote = np.array(
            [
                self._get_proba(
                    nearest_neighbors_idx=nearest_neighbors_idx,
                    class_target=c,
                    weights_arr=weights_arr,
                )
                for c in classes
            ]
        )

        # Return the class label with the maximum probability
        return np.argmax(majority_vote, axis=0)

    def _get_proba(
        self, nearest_neighbors_idx: np.ndarray, class_target: int, weights_arr: np.ndarray
    ) -> np.ndarray:
        """
        Get class probabilities for each input point.

        Parameters
        ----------
        nearest_neighbors_idx : np.ndarray
            Indices of nearest neighbors for each input point.
        class_target : int
            Target class label.
        weights_arr : np.ndarray
            Weights for each input point.

        Returns
        -------
        np.ndarray
            Class probabilities for each input point.
        """
        # Calculate class probabilities based on the presence of neighbors in the target class
        if weights_arr is None:
            return np.mean(self._y[nearest_neighbors_idx] == class_target, axis=1)

        # Calculate weighted class probabilities
        return np.mean(weights_arr * (self._y[nearest_neighbors_idx] == class_target), axis=1)

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
        # Determine the weight function to be applied
        if self.weights == "uniform":
            return None

        # Handling potential division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            weights_arr = 1.0 / np.square(neighbors_distances)

        # Set weights_arr to 1.0 for points with zero distance
        weights_arr[neighbors_distances == 0] = 1.0

        return weights_arr
