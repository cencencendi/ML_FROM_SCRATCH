import numpy as np
from ..tree import DecisionTreeRegressor


def _get_max_features(max_features, n_features):
    """
    Calculate the maximum number of features to consider for splitting.

    Parameters
    ----------
    max_features : int or str
        The maximum number of features to consider. This can be an absolute number,
        "sqrt" for the square root of the total number of features, or "log2" for the
        log base 2 of the total number of features.
    n_features : int
        The total number of features in the dataset.

    Returns
    -------
    int
        The determined maximum number of features to consider.
    """
    if isinstance(max_features, int):
        return max_features
    elif max_features == "sqrt":
        return int(np.sqrt(n_features))
    elif max_features == "log2":
        return int(np.log2(n_features))
    else:
        return n_features


def _get_feature_indices(seed, n_estimators, n_population, n_features):
    """
    Generate indices for a random subset of features for each estimator.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_estimators : int
        Number of estimators in the ensemble.
    n_population : int
        The pool size from which features are drawn.
    n_features : int
        The number of features to draw for each estimator.

    Returns
    -------
    ndarray
        A 2D array where each row contains the indices of the selected features
        for an estimator.
    """
    np.random.seed(seed)
    feature_indices = np.empty((n_estimators, n_features), dtype=int)
    for i in range(n_estimators):
        feature_indices[i] = np.sort(
            np.random.choice(a=n_population, size=n_features, replace=False)
        )
    return feature_indices


def _proba(log_odds: float):
    """
    Convert log odds to probabilities using the logistic function.

    Parameters
    ----------
    log_odds : float
        The log odds value.

    Returns
    -------
    float
        The probability calculated from the log odds.
    """
    return np.exp(log_odds) / (1 + np.exp(log_odds))


class BaseGradientBoosting:
    """
    Base class for gradient boosting models.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    n_estimators : int, optional
        The number of boosting stages to be run.
    criterion : str, optional
        The function to measure the quality of a split.
    max_depth : int, optional
        Maximum depth of the individual regression estimators.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node.
    min_impurity_decrease : float, optional
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    alpha : float, optional
        The alpha-quantile of the huber loss function and the quantile loss function.
    max_features : {int, str}, optional
        The number of features to consider when looking for the best split.
    tol : float, optional
        The tolerance for the termination condition.
    random_state : int, optional
        Controls the randomness of the bootstrapping of the samples used when building trees.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        alpha: float = 0.0,
        max_features=None,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha = alpha
        self.max_features = max_features
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the gradient boosting model.

        Parameters
        ----------
        X : ndarray
            The input samples.
        y : ndarray
            The target values.
        """
        self._X = np.array(X).copy()
        self._y = np.array(y).copy()

        self.n_samples, self.n_features = self._X.shape
        self.estimators = []

        # Determine the maximum number of features to consider for each split
        self.n_max_features = _get_max_features(self.max_features, self.n_features)

        # Generate random feature indices for each estimator
        self.feature_indices = _get_feature_indices(
            self.random_state, self.n_estimators, self.n_features, self.n_max_features
        )

        for estimator_idx in range(self.n_estimators):
            if estimator_idx == 0:
                # Initialize the first estimator as a simple decision stump
                self.estimators.append(DecisionTreeRegressor(max_depth=0))
                self.estimators[estimator_idx].fit(self._X, self._y)
                continue

            # Select features for the current estimator
            x_train = self._X[:, self.feature_indices[estimator_idx]]

            # Compute residuals or pseudo-residuals for boosting
            r_im = self._get_pseudo_residual(y=self._y)

            # Train a new estimator on the residuals
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            self.estimators.append(tree)
            self.estimators[estimator_idx].fit(x_train, r_im)

            # Optionally update terminal regions for specialized loss functions
            if self.update_terminal_regions:
                self.update_terminal_regions(
                    tree=self.estimators[estimator_idx].tree_,
                    terminal_regions=self.estimators[estimator_idx].apply(x_train),
                )

            # Break early if the improvement is below the tolerance
            if np.mean(abs(r_im)) < self.tol:
                break

    def _get_boosted_prediction(self, x: np.ndarray = None):
        """
        Compute the prediction by aggregating the predictions of the base estimators.

        Parameters
        ----------
        x : ndarray, optional
            The input samples.

        Returns
        -------
        ndarray
            The predicted values.
        """
        # Initialize predictions array
        y_preds = np.zeros(shape=(x.shape[0], self.n_estimators))

        # Accumulate predictions from all estimators
        for m in range(len(self.estimators)):
            y_preds[:, m] = self.estimators[m].predict(x[:, self.feature_indices[m]])

        # Aggregate initial prediction and subsequent corrections
        y_pred = y_preds[:, 0] + np.sum(self.learning_rate * y_preds[:, 1:], axis=1)

        return y_pred


class GradientBoostingRegressor(BaseGradientBoosting):
    """
    A gradient boosting regressor.

    This regressor builds an additive model in a forward stage-wise fashion. It allows for the optimization
    of arbitrary differentiable loss functions. In each stage, a regression tree is fit on the negative
    gradient of the given loss function.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    n_estimators : int, optional
        The number of boosting stages to be run.
    criterion : str, optional
        The function to measure the quality of a split.
    max_depth : int, optional
        Maximum depth of the individual regression estimators.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node.
    min_impurity_decrease : float, optional
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    alpha : float, optional
        The alpha-quantile of the huber loss function and the quantile loss function.
    max_features : {int, str}, optional
        The number of features to consider when looking for the best split.
    tol : float, optional
        The tolerance for the termination condition.
    random_state : int, optional
        Controls the randomness of the bootstrapping of the samples used when building trees.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0,
        alpha: float = 0,
        max_features="sqrt",
        tol: float = 0.0001,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            learning_rate,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            alpha,
            max_features,
            tol,
            random_state,
        )
        # Assign the regressor-specific method for calculating pseudo-residuals
        self._get_pseudo_residual = self._get_rim_regressor
        # Placeholder for potential future methods to update terminal regions in trees
        self.update_terminal_regions = None

    def predict(self, X: np.ndarray):
        """
        Predict regression target for each sample in X.

        Parameters
        ----------
        X : ndarray
            Input features array.

        Returns
        -------
        ndarray
            Predicted values.
        """
        # Reshape input for prediction and compute predictions for each sample
        return np.array([self._get_boosted_prediction(x=x.reshape(1, -1)) for x in X]).reshape(-1)

    def _get_rim_regressor(self, y: np.ndarray):
        """
        Calculate the pseudo-residuals for regression.

        Parameters
        ----------
        y : ndarray
            The target values.

        Returns
        -------
        ndarray
            Pseudo-residuals calculated as the difference between target and prediction.
        """
        y_pred = self._get_boosted_prediction(x=self._X)
        return y - y_pred


class GradientBoostingClassifier(BaseGradientBoosting):
    """
    A gradient boosting classifier.

    This classifier builds an additive model in a forward stage-wise fashion. It allows for the optimization
    of arbitrary differentiable loss functions. In each stage, a regression tree is fit on the negative
    gradient of the binomial or multinomial deviance loss function.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    n_estimators : int, optional
        The number of boosting stages to be run.
    criterion : str, optional
        The function to measure the quality of a split.
    max_depth : int, optional
        Maximum depth of the individual regression estimators.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node.
    min_impurity_decrease : float, optional
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    alpha : float, optional
        The alpha-quantile of the huber loss function and the quantile loss function.
    max_features : {int, str}, optional
        The number of features to consider when looking for the best split.
    tol : float, optional
        The tolerance for the termination condition.
    random_state : int, optional
        Controls the randomness of the bootstrapping of the samples used when building trees.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0,
        alpha: float = 0,
        max_features="sqrt",
        tol: float = 0.0001,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            learning_rate,
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            alpha,
            max_features,
            tol,
            random_state,
        )
        # Assign the classifier-specific method for calculating pseudo-residuals
        self._get_pseudo_residual = self._get_rim_classifier

    def _get_rim_classifier(self, y: np.ndarray):
        """
        Calculate the pseudo-residuals for classification using logistic loss.

        Parameters
        ----------
        y : ndarray
            The target class labels.

        Returns
        -------
        ndarray
            Pseudo-residuals for classification.
        """
        # Convert predictions to probabilities and calculate residuals
        y_pred = self._get_boosted_prediction(x=self._X)  # log_odds
        return y - _proba(y_pred)

    def update_terminal_regions(self, tree, terminal_regions, i=0):
        """
        Update the terminal regions of the tree with new values to minimize the loss.

        Parameters
        ----------
        tree : DecisionTreeRegressor
            The decision tree regressor.
        terminal_regions : ndarray
            Indices of the terminal regions for each sample.
        i : int, optional
            Current index in the tree (used for recursive calls).

        """
        if not tree:
            return

        if tree.is_leaf:
            # Update leaf values based on residuals and probabilities
            residual = tree.leaf_values
            y = self._y[terminal_regions == np.array([i])]
            probabilities = y - residual
            tree.value = np.sum(residual) / np.sum((probabilities) * (1 - probabilities))
            tree.leaf_values = None

        # Recursively update children
        self.update_terminal_regions(tree.children_left, terminal_regions, 2 * i + 1)
        self.update_terminal_regions(tree.children_right, terminal_regions, 2 * i + 2)

    def predict(self, X: np.ndarray):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray
            Input features array.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        # Compute predictions and convert to class labels based on threshold
        y_preds = self._get_boosted_prediction(x=X)
        return np.where(_proba(y_preds) < 0.5, 0, 1)
