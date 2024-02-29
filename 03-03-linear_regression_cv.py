import pandas as pd
import numpy as np

from ml_from_scratch.model_selection import cross_val_score, KFold
from ml_from_scratch.linear_model import LinearRegression
from ml_from_scratch.metrics import mean_squarred_error


auto = pd.read_csv("datasets/auto.csv")

X = auto.drop(["mpg"], axis=1)
y = auto["mpg"]

cols_list = [
    ["displacement"],
    ["horsepower"],
    ["weight"],
    ["displacement", "horsepower"],
    ["displacement", "weight"],
    ["horsepower", "weight"],
    ["displacement", "horsepower", "weight"],
]

mse_train_list = []
mse_test_list = []

cv = KFold(n_splits=5, shuffle=True, random_state=42)

for cols in cols_list:
    mse_train_cols, mse_test_cols = cross_val_score(
        estimator=LinearRegression(),
        X=X[cols].to_numpy(),
        y=y.to_numpy(),
        cv=cv,
        scoring="mean_squared_error",
    )

    mse_train_list.append(np.mean(mse_train_cols))
    mse_test_list.append(np.mean(mse_test_cols))

summary = pd.DataFrame(
    {"cols": cols_list, "MSE_training": mse_train_list, "MSE_test": mse_test_list}
)

print("Auto Summary")
print(summary)

best_valid_idx = summary["MSE_test"].argmin()
best_feature = summary.loc[best_valid_idx]["cols"]
print("")
print("=" * 50)
print("")
print(f"Best model features\t: {best_feature}")
print(f'Best valid score\t: {summary["MSE_test"][best_valid_idx]}')
print("")
print("Re-train the best model...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X[best_feature], y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X=X_train.to_numpy(), y=y_train.to_numpy())

pred = model.predict(X_test.to_numpy())

mse = mean_squarred_error(y_true=y_test, y_pred=pred)
print(f"MSE best model \t: {mse}")
