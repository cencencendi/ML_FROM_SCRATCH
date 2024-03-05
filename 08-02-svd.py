from ml_from_scratch.decomposition import SVD
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()

df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])

svd = SVD(n_components=5)
svd.fit(df)
x_transformed = svd.transform(df)

print(svd.explained_variance_ratio_)
print(x_transformed)
