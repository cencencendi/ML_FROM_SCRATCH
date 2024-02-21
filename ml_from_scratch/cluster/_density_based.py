import numpy as np


def _find_neighbors(X: np.ndarray, eps: float):
    """
    Find the neighbors for each point in the dataset within a specified Euclidean distance.

    This function iterates over each point in the dataset, computes the Euclidean distance to all other points,
    and identifies those points lying within a distance less than `eps`. These points are considered neighbors.
    The process leverages a distance computation function to calculate distances between pairs of points.

    Parameters
    ----------
    X : np.ndarray
        The dataset in which neighbors are to be found. It is assumed to be a two-dimensional array
        where each row represents a point in the dataset and columns represent the dimensions of the space.
    eps : float
        The epsilon distance threshold for defining neighbors. Two points are considered neighbors if
        the Euclidean distance between them is less than `eps`.

    Returns
    -------
    list of lists
        A list where each element is a list of indices. Each list of indices represents the neighbors of the
        corresponding point in the dataset. The index in the outer list corresponds to the point in the dataset,
        and its value is a list of indices of its neighbors within `eps` distance.
    """
    neighbors = []  # List to store the neighbors of each point
    # Compute the distance between each pair of points in the dataset
    distance = _compute_distance(x1=X[:, np.newaxis], x2=X)
    # Iterate over the distances to identify neighbors
    for dist in distance:
        # Get indices of points that are within eps distance
        nearest_ids = [idx for idx, val in enumerate(dist) if val < eps]
        neighbors.append(nearest_ids)  # Append the indices of the neighbors

    return neighbors


def _find_core_points(neighbors: list, min_samples: int):
    """
    Identify core points in the dataset. A core point is defined as a point that has at least `min_samples` neighbors within a specified distance (`eps`, not directly used here but implied through the `neighbors` list).

    This function iterates through the list of neighbors for each point in the dataset, provided by the `neighbors` parameter. A point is considered a core point if the number of its neighbors (excluding itself) is at least equal to `min_samples`. This concept is fundamental in density-based clustering algorithms like DBSCAN, where core points are used to form the basis of clusters.

    Parameters
    ----------
    neighbors : list of lists
        A list where each element is a list of indices representing the neighbors of each point in the dataset. The index in the outer list corresponds to a specific point in the dataset, and its value is a list of indices of its neighbors.
    min_samples : int
        The minimum number of neighbors a point must have to be considered a core point. This parameter controls the density threshold needed to form a cluster.

    Returns
    -------
    list
        A list of indices of the points in the dataset that are considered core points. These points have at least `min_samples` neighbors.
    """
    # Core points are those with at least min_samples neighbors
    return [idx for idx, val in enumerate(neighbors) if len(val) >= min_samples]


def _compute_distance(x1: np.ndarray, x2: np.ndarray):
    """
    Compute the Euclidean distance between each pair of points represented by the numpy arrays x1 and x2.

    The function calculates the pairwise Euclidean distance between points in `x1` and `x2`. It's designed to
    handle multidimensional data, where each point is a row in the input arrays. The Euclidean distance is
    the square root of the sum of the squared differences between corresponding elements of the points.

    This function is a fundamental component of many clustering algorithms, including DBSCAN, where distances
    between points are used to determine neighborhood relationships.

    Parameters
    ----------
    x1 : np.ndarray
        A numpy array where each row represents a point in the dataset, and columns represent the dimensions
        of the space. This array represents the 'source' points from which distances are measured.
    x2 : np.ndarray
        A numpy array similar to `x1`, representing the 'target' points to which distances are measured.
        The function computes the distance from each point in `x1` to each point in `x2`.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each element (i, j) represents the Euclidean distance between the ith point
        in `x1` and the jth point in `x2`. Thus, the output is a matrix of distances with shape
        (num_points_x1, num_points_x2).

    Notes
    -----
    The computation is vectorized for efficiency, leveraging numpy's broadcasting and aggregation capabilities
    to calculate distances without explicit loops over the points. This approach significantly improves the
    performance for large datasets.
    """
    # The Euclidean distance is calculated and returned
    return np.sqrt(((x1 - x2) ** 2).sum(axis=2))


class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        """
        Initialize the DBSCAN clustering algorithm instance with specified parameters.

        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that
        creates clusters based on the density of points in a space. It identifies clusters of varying shapes
        in noisy data sets by considering core points, border points, and noise.

        Parameters
        ----------
        eps : float, optional
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster. It is the most important DBSCAN
            parameter to choose appropriately for your data set and distance function.
        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself.
        """
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X: np.ndarray) -> None:
        """
        Apply the DBSCAN clustering algorithm to the dataset X.

        This method finds core points, expands clusters from them, and assigns cluster labels to points. Points that
        do not belong to any cluster are marked as noise.

        Parameters
        ----------
        X : np.ndarray
            The input data to cluster. It should be a two-dimensional array of shape (n_samples, n_features),
            where each row represents a single data point.

        Returns
        -------
        None
            This method does not return a value but it updates the instance's state with the clustering results,
            accessible via the `assignment` attribute, which maps cluster IDs to the indices of the points belonging
            to those clusters. Noise points are assigned to cluster ID -1.
        """
        # Copy the input dataset and start the clustering process
        self._X = np.array(X).copy()

        # Find neighbors for each point in the dataset
        neighbors = _find_neighbors(X=self._X, eps=self.eps)
        # Identify core points
        core_ind = _find_core_points(neighbors=neighbors, min_samples=self.min_samples)

        # Initialize cluster assignments and visited points set
        self.assignment, self.visited = {}, set()
        next_cluster_id = 0  # ID for the next cluster

        # Expand clusters starting from each core point
        for i in core_ind:
            if i in self.visited:
                continue  # Skip if the point is already visited

            # Expand cluster from the current core point
            self._expand_cluster(
                p=i, neighbors=neighbors, core_ind=core_ind, next_cluster_id=next_cluster_id
            )
            self.visited.add(i)  # Mark the core point as visited
            next_cluster_id += 1  # Increment the cluster ID for the next cluster

        # Handle noise points (points not belonging to any cluster)
        noise_cluster = []
        for idx, point in enumerate(self._X):
            if idx not in self.visited:
                noise_cluster.append(idx)  # Add unvisited points to noise cluster

        # Assign noise points to a special cluster with an ID of -1
        if noise_cluster:
            self.assignment[-1] = noise_cluster

    def _expand_cluster(self, p, neighbors, core_ind, next_cluster_id):
        """
        Expand the cluster from a core point by recursively adding all directly density-reachable points to the cluster.

        This method is a helper function used during the `fit` process to expand the clusters from each core point. It
        marks all points within `eps` distance of core points as part of the cluster, including other core points.

        Parameters
        ----------
        p : int
            The index of the core point from which to start expanding the cluster.
        neighbors : list of lists
            The list of neighbor lists for each point in the dataset. Each list contains the indices of other points
            that are within `eps` distance.
        core_ind : list
            The list of indices of core points in the dataset.
        next_cluster_id : int
            The ID to assign to the newly formed cluster.

        Returns
        -------
        None
            This method does not return a value but directly updates the `assignment` attribute of the instance,
            adding points to the cluster with ID `next_cluster_id`.
        """
        # Initialize reachable set with neighbors of the core point
        reachable = set(neighbors[p])
        # List of points to visit during cluster expansion
        points_to_visit = list(reachable)

        while points_to_visit:

            # Pop the first element to simulate queue behavior
            point_idx = points_to_visit.pop(0)

            # Skip if already visited
            if point_idx not in self.visited:
                continue

            self.visited.add(point_idx)

            if point_idx in core_ind:
                # Ensure we only add unvisited points
                new_points = set(neighbors[point_idx]) - self.visited
                # Update reachable with new points
                reachable.update(new_points)
                # Add new points to points_to_visit for further exploration
                points_to_visit.extend(new_points)

        self.assignment[next_cluster_id] = list(reachable)
