# Importing necessary libraries
import copy
import numpy as np


# Function to generate possible split points for a given array of data
def _generate_possible_threshold(data: np.ndarray) -> list:
    """
    Generates possible split points for a given array of data.

    Parameters
    ----------
    data : np.ndarray
        Array of numerical values for which split points are generated.

    Returns
    -------
    list
        List of possible split points.
    """
    return [0.5 * (data[i] + data[i + 1]) for i in range(len(np.unique(data)) - 1)]


def _generate_sample_indices(n_population, n_samples, weights):
    """
    Generates random sample indices based on weights.

    Parameters
    ----------
    n_population : int
        Total number of items in the population.
    n_samples : int
        Number of samples to be generated.
    weights : array-like
        Weights assigned to each item in the population.

    Returns
    -------
    np.ndarray
        Array of randomly selected sample indices.
    """
    return np.random.choice(a=n_population, size=n_samples, p=weights)


class DecisionStump:
    def __init__(self, feature_idx: int = None, threshold: float = None, polarity: int = 1) -> None:
        """
        Initialize a DecisionStump instance.

        Parameters
        ----------
        feature_idx : int, optional
            Index of the feature used for splitting, by default None.
        threshold : float, optional
            Threshold value for the split, by default None.
        polarity : int, optional
            Polarity of the decision stump, by default 1.
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity

    def _best_split(self, weights: np.ndarray):
        """
        Find the best split for the decision stump based on weighted errors.

        Parameters
        ----------
        weights : np.ndarray
            Weights assigned to each sample.

        Returns
        -------
        tuple
            Best feature index, threshold, and polarity.
        """
        # Initialize variables to store the best split configuration
        best_feature, best_threshold, best_polarity = None, None, None
        min_error = float("inf")  # Initialize minimum error to positive infinity

        # Iterate over each feature index
        for feat_idx in range(self.n_features):
            # Generate possible threshold values for the current feature
            thresholds = _generate_possible_threshold(data=self._X[:, feat_idx])

            # Iterate over each threshold value
            for thr in thresholds:
                polarity = 1  # Initialize polarity to positive
                predictions = np.ones(self.n_samples) * polarity
                predictions[self._X[:, feat_idx] <= thr] = (
                    -1
                )  # Update predictions based on the threshold

                # Calculate the weighted error using the specified weights
                weighted_error = np.dot(weights, self._y != predictions)

                # Adjust polarity and error if the weighted error is above 0.5
                if weighted_error > 0.5:
                    weighted_error = 1 - weighted_error
                    polarity = -1

                # Update best split configuration if current error is smaller
                if weighted_error < min_error:
                    best_feature = feat_idx
                    best_threshold = thr
                    best_polarity = polarity
                    min_error = weighted_error

        # Return the best split configuration
        return best_feature, best_threshold, best_polarity

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        """
        Fit the decision stump to the given data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target labels.
        weights : np.ndarray
            Weights assigned to each sample.
        """
        # Copy input data and labels to instance variables
        self._X = np.array(X).copy()
        self._y = np.array(y).copy()

        # Get the number of samples and features in the input data
        self.n_samples, self.n_features = self._X.shape

        # Find the best split using the provided weights
        self.feature_idx, self.threshold, self.polarity = self._best_split(weights)

    def predict(self, X: np.ndarray):
        """
        Make predictions using the fitted decision stump.

        Parameters
        ----------
        X : np.ndarray
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        # Copy input features to a new array
        X_test = np.array(X).copy()

        # Initialize an array for predicted labels with the chosen polarity
        y_pred = np.ones(shape=X_test.shape[0]) * self.polarity

        # Update predicted labels based on the split condition
        if self.polarity == 1:
            y_pred[X_test[:, self.feature_idx] <= self.threshold] = -1
        else:
            y_pred[X_test[:, self.feature_idx] <= self.threshold] = 1

        # Return the predicted labels
        return y_pred


class AdaBoostClassifier:
    def __init__(self, estimator=None, n_estimator: int = 5, weighted_sample: bool = True) -> None:
        """
        Initialize an AdaBoostClassifier instance.

        Parameters
        ----------
        estimator : object, optional
            Base estimator for boosting, by default None.
        n_estimator : int, optional
            Number of boosting rounds, by default 5.
        weighted_sample : bool, optional
            Whether to use weighted sampling, by default True.
        """
        self.estimator = estimator
        self.n_estimator = n_estimator
        self.weighted_sample = weighted_sample

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the AdaBoostClassifier to the given data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target labels.
        """
        # If base estimator is not provided, use DecisionStump as the default
        if self.estimator is None:
            base_estimator = DecisionStump()

        # Copy input data and labels to instance variables
        self._X = np.array(X).copy()
        self._y = np.array(y).copy()

        # Get the number of samples and features in the input data
        self.n_samples, self.n_features = self._X.shape

        # Initialize weights for each sample (initially uniform)
        self.weights = np.ones(self.n_samples) / self.n_samples
        self.alpha = np.zeros(self.n_estimator)  # Initialize alpha values
        self.estimators = np.empty(
            shape=self.n_estimator, dtype=object
        )  # List to store the trained estimators

        # Iterate over the specified number of boosting rounds
        for i in range(self.n_estimator):
            # If weighted sampling is enabled, select samples based on weights
            if self.weighted_sample:
                sample_ids = _generate_sample_indices(
                    n_population=self.n_samples, n_samples=self.n_samples, weights=self.weights
                )

                X_train = self._X[sample_ids, :]
                y_train = self._y[sample_ids]
            else:
                X_train = self._X
                y_train = self._y

            # Create a deep copy of the base estimator and fit it to the current data
            estimator = copy.deepcopy(base_estimator)
            estimator.fit(X_train, y_train, weights=self.weights)

            # Make predictions and calculate weighted error
            y_pred = estimator.predict(X=X_train)
            weighted_error = np.dot(self.weights, y_train != y_pred)

            # Calculate the contribution of the current estimator to the final prediction
            alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))

            # Update weights and normalize them
            self.weights *= np.exp(-alpha * y_train * y_pred)
            self.weights /= np.sum(self.weights)

            # Store the trained estimator and its contribution in the lists
            self.estimators[i] = estimator
            self.alpha[i] = alpha

    def predict(self, X: np.ndarray):
        """
        Make predictions using the fitted AdaBoostClassifier.

        Parameters
        ----------
        X : np.ndarray
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        # Initialize an array for predicted labels with zeros
        y_pred = np.zeros(shape=X.shape[0])

        # Iterate over trained estimators and their corresponding alpha values
        for estimator, alpha in zip(self.estimators, self.alpha):
            # Get predictions from the current estimator
            preds = estimator.predict(X)

            # Update the ensemble prediction by adding the weighted contribution
            y_pred += alpha * preds

        # Apply sign function to get the final predicted labels (binary classification)
        return np.sign(y_pred)
