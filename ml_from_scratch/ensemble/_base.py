import numpy as np
import copy


# Function to create a list of deep copies of the given estimator
def _get_ensemble_estimators(estimator, n_estimators):
    """
    Creates a list of deep copies of the provided estimator.

    Parameters
    ----------
    estimator : object
        The base estimator to be copied.
    n_estimators : int
        The number of copies to create.

    Returns
    -------
    list
        A list containing deep copies of the provided estimator.
    """
    return [copy.deepcopy(estimator) for _ in range(n_estimators)]


# Function to generate random sample indices for bootstrapping
def _generate_sample_indices(seed, n_estimators, n_population, n_samples):
    """
    Generates random sample indices for bootstrapping.

    Parameters
    ----------
    seed : int
        Seed for reproducibility.
    n_estimators : int
        The number of estimators (copies).
    n_population : int
        Size of the population to sample from.
    n_samples : int
        Number of samples to generate for each estimator.

    Returns
    -------
    ndarray
        A NumPy array containing random sample indices for bootstrapping.
    """
    np.random.seed(seed)
    return np.random.choice(a=n_population, size=(n_estimators, n_samples), replace=True)


# Function to determine the maximum number of features to consider
def _get_max_features(max_features, n_features):
    """
    Determines the maximum number of features to consider based on the input.

    Parameters
    ----------
    max_features : int, str
        Maximum number of features to consider.
    n_features : int
        Total number of features in the dataset.

    Returns
    -------
    int
        An integer representing the maximum number of features to consider.
    """
    if isinstance(max_features, int):
        return max_features
    if max_features == "sqrt":
        return int(np.sqrt(n_features))
    if max_features == "log2":
        return int(np.log2(n_features))
    return n_features


# Function to generate random feature indices for subsetting
def _get_feature_indices(seed, n_estimators, n_population, n_features):
    """
    Generates random feature indices for subsetting.

    Parameters
    ----------
    seed : int
        Seed for reproducibility.
    n_estimators : int
        The number of estimators (copies).
    n_population : int
        Size of the population to sample features from.
    n_features : int
        Total number of features in the dataset.

    Returns
    -------
    ndarray
        A 2D NumPy array containing random feature indices for subsetting.
    """
    np.random.seed(seed)
    feature_indices = np.empty((n_estimators, n_features), dtype=int)
    for i in range(n_estimators):
        feature_indices[i] = np.sort(
            np.random.choice(a=n_population, size=n_features, replace=False)
        )
    return feature_indices


# Function to make predictions using an ensemble of estimators
def _predict_ensemble(estimators, features, X):
    """
    Makes predictions using an ensemble of estimators on the provided features and data.

    Parameters
    ----------
    estimators : list
        List of base estimators in the ensemble.
    features : ndarray
        Indices of features to use for each estimator.
    X : ndarray
        Input data for making predictions.

    Returns
    -------
    ndarray
        A 2D NumPy array containing predictions from each estimator in the ensemble.
    """
    n_samples, _ = X.shape

    y_pred_ensemble = np.empty(shape=(len(estimators), n_samples))
    for b, estimator in enumerate(estimators):
        X_ = X[:, features[b]]
        y_pred_ensemble[b] = estimator.predict(X_)

    return y_pred_ensemble


class BaseEnsemble:
    def __init__(
        self, estimator, n_estimators: int, max_features=None, random_state: int = 42
    ) -> None:
        """
        Initializes the BaseEnsemble instance.

        Parameters
        ----------
        estimator : object
            The base estimator to be used in the ensemble.
        n_estimators : int
            The number of estimators (copies) in the ensemble.
        max_features : int, str, or None, optional
            The maximum number of features to consider for each estimator.
            If None, all features will be used. If int or str, it will be applied uniformly across estimators.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        None
        """
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the ensemble by training each individual estimator on bootstrapped and subsetted data.

        Parameters
        ----------
        X : ndarray
            Input data for training.
        y : ndarray
            Target values for training.

        Returns
        -------
        None
        """
        self._X = np.array(X).copy()
        self._y = np.array(y).copy()

        self.n_samples, self.n_features = self._X.shape

        # Create ensemble estimators
        self.ensemble_estimators = _get_ensemble_estimators(
            estimator=self.estimator, n_estimators=self.n_estimators
        )

        # Generate random sample indices for bootstrapping
        sample_indices = _generate_sample_indices(
            seed=self.random_state,
            n_estimators=self.n_estimators,
            n_population=self.n_samples,
            n_samples=self.n_samples,
        )

        # Determine the maximum number of features to consider
        max_features = _get_max_features(max_features=self.max_features, n_features=self.n_features)

        # Generate random feature indices for subsetting
        self.feature_indices = _get_feature_indices(
            seed=self.random_state,
            n_estimators=self.n_estimators,
            n_population=self.n_features,
            n_features=max_features,
        )

        # Train each estimator on bootstrapped and subsetted data
        for b, estimator in enumerate(self.ensemble_estimators):
            X_bootstrap = self._X[:, self.feature_indices[b]]
            X_bootstrap = X_bootstrap[sample_indices[b], :]

            y_bootstrap = self._y[sample_indices[b]]

            estimator.fit(X_bootstrap, y_bootstrap)

    def predict(self, X: np.ndarray):
        """
        Makes predictions using the ensemble of trained estimators.

        Parameters
        ----------
        X : ndarray
            Input data for making predictions.

        Returns
        -------
        ndarray
            Predictions from the ensemble.
        """
        X_test = np.array(X).copy()

        # Make predictions using the ensemble
        y_pred_ensemble = _predict_ensemble(
            estimators=self.ensemble_estimators, features=self.feature_indices, X=X_test
        )

        # Aggregate predictions using the specified function
        return self.agg_func(y_pred_ensemble)
