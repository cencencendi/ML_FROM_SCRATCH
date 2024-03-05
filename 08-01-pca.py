from ml_from_scratch.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()

df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])

pca = PCA(n_components=5)
pca.fit(df)
x_transformed = pca.transform(df)

print(pca.explained_variance_ratio_)
print(x_transformed)
