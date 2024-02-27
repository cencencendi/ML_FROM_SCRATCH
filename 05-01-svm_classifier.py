import numpy as np
from ml_from_scratch.svm import SVC

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, -1, -1])

clf = SVC()
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))
