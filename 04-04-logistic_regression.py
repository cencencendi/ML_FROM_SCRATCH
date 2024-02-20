from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression

from ml_from_scratch.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy

# Load the Iris dataset
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(
    -1, 1
)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=10000, alpha=0.1)

# Train the model on the training data
logistic_regression_model.fit(X, y)

# Make predictions on the test set
y_pred = logistic_regression_model.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
classification_report_str = classification_report(y, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_str)
