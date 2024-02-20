import numpy as np


class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None) -> None:
        """
        Initialize the KFold cross-validation splitter.

        Parameters
        ----------
        n_splits : int, optional
            Number of splits/folds. Default is 5.
        shuffle : bool, optional
            Whether to shuffle the data before splitting. Default is False.
        random_state : int, optional
            Seed for the random number generator. Default is None.

        Returns
        -------
        None
        """
        # Check and set input parameters
        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more."
            )
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))
        if not shuffle and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True."
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(random_state)

    def split(self, X: np.ndarray = None):
        """
        Generate indices to split data into training and testing sets.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Yields
        ------
        tuple
            Tuple of arrays (train_indices, test_indices) for each split.
        """
        # Ensure input data is a NumPy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Get the number of samples and create an array of indices
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # Shuffle the indices if specified
        if self.shuffle:
            self.random_state.shuffle(indices)

        # Determine the sizes of each fold
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0

        # Generate train/test indices for each fold
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield (np.concatenate([indices[:start], indices[stop:]]), indices[start:stop])
            current = stop
