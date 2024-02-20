# Import necessary libraries
import numpy as np
from . import _criterion  # Importing criterion module from the same package
from collections import Counter  # Import Counter class from collections module

# Define a dictionary mapping criterion names to corresponding criterion classes for classification
CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "entropy": _criterion.Entropy,
    "log_loss": _criterion.LogLoss,
}

# Define a dictionary mapping criterion names to corresponding criterion classes for regression
CRITERIA_REG = {"mse": _criterion.MSE, "mae": _criterion.MAE}


# Function to calculate the majority vote from an array of labels
def _calculate_majority_vote(y: np.ndarray = None) -> int:
    """
    Calculates the majority vote from an array of labels.

    Parameters
    ----------
    y : np.ndarray
        Array of labels.

    Returns
    -------
    int
        The label with the highest count (majority vote).
    """
    return Counter(y).most_common(1)[0][0]


# Function to calculate the mean vote from an array of numerical values
def _calculate_mean_vote(y: np.ndarray = None) -> float:
    """
    Calculates the mean vote from an array of numerical values.

    Parameters
    ----------
    y : np.ndarray
        Array of numerical values.

    Returns
    -------
    float
        Mean of the numerical values.
    """
    return np.mean(y)


# Function to split data into left and right based on a given threshold
def _split_data(data: np.ndarray, threshold: float) -> tuple:
    """
    Splits data into left and right based on a given threshold.

    Parameters
    ----------
    data        : np.ndarray
        Array of numerical values to be split.

    threshold   : float
        Threshold value for splitting.

    Returns
    -------
    tuple
        Two arrays of indices representing the left and right splits.
    """
    left_ids = np.argwhere(data <= threshold).flatten()
    right_ids = np.argwhere(data > threshold).flatten()
    return left_ids, right_ids


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


class Tree:
    def __init__(
        self,
        feature: str = None,
        threshold: float = None,
        value: float = None,
        impurity: float = None,
        children_left=None,
        children_right=None,
        is_leaf: bool = False,
        n_samples: int = None,
        leaf_values: np.ndarray = None,
    ) -> None:
        """
        Constructor for the Tree class.

        Parameters
        ----------
        feature     : str
            The feature used for splitting at this node.
        threshold   : float
            The threshold value for the split.
        value       : float
            The predicted value at this leaf node for regression tasks.
        impurity    : float
            The impurity measure (e.g., Gini index, entropy) at this node.
        children_left:
            The left child node of the current node.
        children_right:
            The right child node of the current node.
        is_leaf     : bool
            Indicates whether the node is a leaf.
        n_samples   : int
            The number of samples in the node.

        Returns
        -------
        None
        """
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.impurity = impurity
        self.children_left = children_left
        self.children_right = children_right
        self.is_leaf = is_leaf
        self.n_samples = n_samples
        self.leaf_values = leaf_values


class BaseDecisionTree:
    def __init__(
        self,
        criterion: str,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        alpha: float = 0.0,
    ) -> None:
        """
        Constructor for the BaseDecisionTree class.

        Parameters
        ----------
        criterion : str
            The criterion used for splitting nodes (e.g., "gini", "entropy", "mse").
        max_depth : int
            The maximum depth of the decision tree.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node.
        min_impurity_decrease : float
            Minimum impurity decrease required for a split to happen.
        alpha : float
            Regularization parameter for pruning (default is 0.0).

        Returns
        -------
        None
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the decision tree to the given training data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target labels.

        Returns
        -------
        None
        """
        # Copy the input features and target labels to the instance variables
        self._X = np.copy(X)
        self._y = np.copy(y)

        # Get the number of samples and features in the input data
        self.n_samples, self.n_features = self._X.shape

        # Grow the decision tree using the input data
        self.tree_ = self._grow_tree(self._X, self._y)

        self._prune_tree()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        # Predict labels for each input feature using the trained decision tree
        return np.array([self._traverse_tree(x)[0] for x in X])

    def _traverse_tree(self, x: np.ndarray, tree: Tree = None, i: int = 0) -> float:
        """
        Recursively traverse the decision tree to make predictions.

        Parameters
        ----------
        x    : np.ndarray
            Input feature vector.
        tree : Tree
            Current node in the decision tree.

        Returns
        -------
        float or int
            Prediction value.
        """
        # If no specific tree is provided, use the main tree
        if tree is None:
            tree = self.tree_

        # Check if the current node is a leaf, return the value if true
        if tree.is_leaf:
            return tree.value, i

        # Traverse the left or right child based on the feature threshold
        if x[tree.feature] <= tree.threshold:
            return self._traverse_tree(x, tree.children_left, 2 * i + 1)
        else:
            return self._traverse_tree(x, tree.children_right, 2 * i + 2)

    def apply(self, X: np.ndarray):
        """
        Apply the trained decision tree model to make predictions on a dataset.

        Parameters
        ----------
        X : np.ndarray
            The input dataset for which predictions are to be made. It should be a 2D array
            where each row represents a single sample and each column represents a feature.

        Returns
        -------
        np.ndarray
            An array of integers, where each integer corresponds to the index of the leaf node
            in the decision tree that each sample in X is associated with. This can be used
            to identify the path taken by each sample in the decision tree.
        """
        return np.array([self._traverse_tree(x)[1] for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Tree:
        """
        Recursively grow the decision tree.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target labels.
        depth : int
            Current depth of the tree.

        Returns
        -------
        Tree
            A node in the decision tree.
        """
        # Calculate impurity and value for the current node
        node_impurity = self._impurity_evaluation(y)
        node_value = self._leaf_value_calculation(y)

        # Create a leaf node if max depth is reached or no more splits
        node = Tree(
            value=node_value, impurity=node_impurity, is_leaf=True, n_samples=len(y), leaf_values=y
        )

        # Recursively split data and grow tree if conditions are met
        if self.max_depth is None or depth < self.max_depth:
            best_feature, best_threshold = self._best_split(X, y)

            if best_feature is not None:
                # Split data based on the best feature and threshold
                left_ids, right_ids = _split_data(data=X[:, best_feature], threshold=best_threshold)

                X_left, X_right = X[left_ids], X[right_ids]
                y_left, y_right = y[left_ids], y[right_ids]

                # Update node properties and continue growing the tree
                node.feature = best_feature
                node.threshold = best_threshold
                node.children_left = self._grow_tree(X=X_left, y=y_left, depth=depth + 1)
                node.children_right = self._grow_tree(X=X_right, y=y_right, depth=depth + 1)
                node.is_leaf = False
                node.leaf_values = None

        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best split for a node.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target labels.

        Returns
        -------
        Tuple
            Best feature index and best threshold for the split.
        """
        # Check if the number of samples is below the minimum required for a split
        if len(y) < self.min_samples_split:
            return None, None

        # Initialize variables for tracking the best split
        best_gain = 0.0
        best_feature, best_threshold = None, None

        # Iterate through features and thresholds to find the best split
        for feat_i in range(self.n_features):
            thresholds = _generate_possible_threshold(data=X[:, feat_i])

            for threshold in thresholds:
                # Split data based on the current feature and threshold
                left_ids, right_ids = _split_data(data=X[:, feat_i], threshold=threshold)

                y_left, y_right = y[left_ids], y[right_ids]

                # Check if the minimum samples condition is met for both child nodes
                if all([len(y_left), len(y_right)]) >= self.min_samples_leaf:
                    # Calculate impurity decrease for the current split
                    current_gain = self._calculate_impurity_decrease(y, y_left, y_right)

                    # Update best split if the current split provides higher impurity decrease
                    if current_gain > best_gain:
                        best_gain = current_gain
                        best_feature = feat_i
                        best_threshold = threshold

        # Check if the best gain satisfies the minimum impurity decrease condition
        if best_gain >= self.min_impurity_decrease:
            return best_feature, best_threshold

        return None, None

    def _calculate_impurity_decrease(
        self, parent: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> float:
        """
        Calculate the impurity decrease for a potential split.

        Parameters
        ----------
        parent : np.ndarray
            Labels of the parent node.
        left : np.ndarray
            Labels of the left child node.
        right : np.ndarray
            Labels of the right child node.

        Returns
        -------
        float
            Impurity decrease.
        """
        # Calculate impurities for parent, left, and right nodes
        N_T = len(parent)
        N_t_L = len(left)
        N_t_R = len(right)

        parent_impurity = self._impurity_evaluation(parent)
        left_child_impurity = self._impurity_evaluation(left)
        right_child_impurity = self._impurity_evaluation(right)

        # Calculate and return the impurity decrease for the potential split
        impurity_decrease = (N_T / self.n_samples) * (
            parent_impurity
            - (N_t_L / N_T) * left_child_impurity
            - (N_t_R / N_T) * right_child_impurity
        )

        return impurity_decrease

    def _prune_tree(self, tree: Tree = None) -> None:
        """
        Prune grown tree to avoid overfitting.

        Parameters
        ----------
        tree : Tree
            The tree or the branch that will be pruned.

        Returns
        -------
        None
        """
        # If no specific tree is provided, use the main tree
        if tree is None:
            tree = self.tree_

        # Check if we've down to the leaf, then skip
        if tree.is_leaf:
            pass
        else:
            # Recursively check trough the children
            self._prune_tree(tree.children_left)
            self._prune_tree(tree.children_right)

            # Check if the children are not a leaf (is_leaf = False)
            if not tree.children_left.is_leaf and not tree.children_right.is_leaf:
                # Determine the number of samples in the children
                n_left = tree.children_left.n_samples
                n_right = tree.children_right.n_samples

                # Calculate the probability
                p = n_left / (n_left + n_right)

                # Calculate the delta (impurity decrease)
                delta = (
                    tree.impurity
                    - p * tree.children_left.impurity
                    - (1 - p) * tree.children_right.impurity
                )

                # If the impurity of the tree or branch is not significantly decrease, then prune it.
                if delta < self.alpha:
                    tree.children_left, tree.children_right = None, None
                    tree.threshold = None
                    tree.feature = None
                    tree.is_leaf = True

    def _print_tree(self, tree: Tree = None) -> None:
        """
        Print a textual representation of the decision tree structure.

        Parameters
        ----------
        tree : Tree, optional
            The tree or branch to be printed. If not provided, the main tree is used.

        Returns
        -------
        None
        """
        # If no specific tree is provided, use the main tree
        if tree is None:
            tree = self.tree_

        def _to_string(tree: Tree, indent: str = "|  ") -> str:
            """
            Recursively convert the tree structure to a string for printing.

            Parameters
            ----------
            tree : Tree
                The current tree or branch being processed.
            indent : str
                The indentation string for better visualization.

            Returns
            -------
            str
                A string representation of the current tree or branch.
            """
            # If it's a leaf node, print the prediction value
            if tree.is_leaf:
                return f"Prediction: {tree.value:.2f}"

            # If it's not a leaf node, print the decision condition
            show_thres = f"Is feature-{tree.feature} <= {tree.threshold:.2f}?"
            # Recursively print the left and right branches
            left_branch = f'{indent}T: {_to_string(tree.children_left, indent= indent + "|  ")}'
            right_branch = f'{indent}F: {_to_string(tree.children_right, indent= indent + "|  ")}'
            return f"{show_thres}\n{left_branch}\n{right_branch}"

        # Print the header
        print("The Decision Tree")
        print("=" * 50)
        # Print the tree structure recursively
        print(_to_string(tree=self.tree_))


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        alpha: float = 0,
    ) -> None:
        """
        Constructor for the DecisionTreeClassifier class, inheriting from BaseDecisionTree.

        Parameters
        ----------
        criterion : str
            The criterion used for splitting nodes (e.g., "gini", "entropy", "log_loss").
        max_depth : int
            The maximum depth of the decision tree.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node.
        min_impurity_decrease : float
            Minimum impurity decrease required for a split to happen.
        alpha : float
            Regularization parameter for pruning (default is 0).

        Returns
        -------
        None
        """
        super().__init__(
            criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, alpha
        )
        # Set impurity evaluation function and leaf value calculation method for classification
        self._impurity_evaluation = CRITERIA_CLF[self.criterion]
        self._leaf_value_calculation = _calculate_majority_vote

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the decision tree classifier to the given training data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target labels.

        Returns
        -------
        None
        """
        super(DecisionTreeClassifier, self).fit(X, y)


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(
        self,
        criterion: str = "mse",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        alpha: float = 0,
    ) -> None:
        """
        Constructor for the DecisionTreeRegressor class, inheriting from BaseDecisionTree.

        Parameters
        ----------
        criterion : str
            The criterion used for splitting nodes (e.g., "mse", "mae").
        max_depth : int
            The maximum depth of the decision tree.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node.
        min_impurity_decrease : float
            Minimum impurity decrease required for a split to happen.
        alpha : float
            Regularization parameter for pruning (default is 0).

        Returns
        -------
        None
        """
        super().__init__(
            criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, alpha
        )
        # Set impurity evaluation function and leaf value calculation method for regression
        self._impurity_evaluation = CRITERIA_REG[self.criterion]
        self._leaf_value_calculation = _calculate_mean_vote

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the decision tree regressor to the given training data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target labels.

        Returns
        -------
        None
        """
        super(DecisionTreeRegressor, self).fit(X, y)
