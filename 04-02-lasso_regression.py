import pandas as pd

# import numpy as np

# from ml_from_scratch.model_selection import cross_val_score
from ml_from_scratch.linear_model import Lasso

# from ml_from_scratch.metrics import mean_squarred_error


auto = pd.read_csv("datasets/auto.csv")

X = auto.drop(["mpg"], axis=1)
y = auto["mpg"]

lasso = Lasso(alpha=0.1, tol=1e-4, max_iter=100)
lasso.fit(X.to_numpy(), y.to_numpy())
