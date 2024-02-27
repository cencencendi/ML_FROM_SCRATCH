import pandas as pd
import matplotlib.pyplot as plt
from ml_from_scratch.neighbors import KNeighborsRegressor
import numpy as np

df = pd.read_csv("datasets/bmd.csv")

X = df.age.to_numpy().reshape(-1, 1)
y = df.bmd.to_numpy()

knn = KNeighborsRegressor(n_neighbor=80)
knn.fit(X, y)
pred = knn.predict(np.sort(X, axis=0))

plt.scatter(X, y)
plt.plot(np.sort(X, axis=0), pred)
plt.show()
