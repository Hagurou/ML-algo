import numpy as np
from sklearn.datasets import load_digits


class PCA:
    def __init__(self, topk):
        self.mean = None
        self.components = None
        self.topk = topk

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        x_norm = X - self.mean
        cov_matrix = np.cov(x_norm, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sort_cond = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sort_cond]
        eigenvectors = eigenvectors[:, sort_cond]
        self.components = eigenvectors[:, :self.topk]

    def transform(self, X):
        x_norm = X - self.mean
        return np.dot(x_norm, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def variance_retained(self):
        total_variance = np.sum(self.eigenvalues)
        explained_variance = np.sum(self.eigenvalues[:self.topk]) / total_variance
        return explained_variance


X = load_digits().data
pca = PCA(topk=30)
X_pca = pca.fit_transform(X)
print("Before PCA: ", X.shape)
print("After PCA: ", X_pca.shape)
print(f"Variance retained: {100 * pca.variance_retained():.2f}%")
