from ml_from_scratch.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from ml_from_scratch.metrics import mean_squarred_error

X, y = make_regression(n_samples=1000, n_features=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=5)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

print(f"MSE: {mean_squarred_error(y_test, pred)}")
