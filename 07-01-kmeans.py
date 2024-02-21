from ml_from_scratch.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

k = 10
X = np.random.rand(1000, 2) * 10

kmeans = KMeans(n_clusters=k)
kmeans.fit(X)


plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, label="Data Points")

# Plot centroids
plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    c="red",
    s=100,
    marker="X",
    label="Centroids",
)

plt.title("Data Points and Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
