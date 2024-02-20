from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml_from_scratch.ensemble import BaggingClassifier
from ml_from_scratch.metrics import accuracy_score

# Generate dummy data
X, y = make_classification(
    n_samples=1000,  # Number of samples
    n_features=20,  # Number of features
    random_state=42,
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = BaggingClassifier(n_estimators=100)
clf.fit(X_train, y_train)

pred = clf.predict(X=X_test)

print(f"Accuracy: {(accuracy_score(y_true=y_test, y_pred=pred))*100}%")
