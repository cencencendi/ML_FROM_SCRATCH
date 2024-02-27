import numpy as np
from ml_from_scratch.cluster import DBSCAN

np.random.seed(42)
X = np.random.rand(20, 2) * 5

db = DBSCAN(eps=1.3)
db.fit(X)

print(db.assignment)
