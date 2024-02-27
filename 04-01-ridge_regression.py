import pandas as pd

# import numpy as np

# from ml_from_scratch.model_selection import cross_val_score
from ml_from_scratch.linear_model import Ridge

# from ml_from_scratch.metrics import mean_squarred_error


auto = pd.read_csv("datasets/auto.csv")

X = auto.drop(["mpg"], axis=1)
y = auto["mpg"]

ridge = Ridge(alpha=4)
ridge.fit(X, y)
