from ml_from_scratch.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml_from_scratch.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=4)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred)*100}%")
