import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GBCFromScratch:

    def __init__(self, learning_rate, n_estimators):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        n = X.shape[0]
        f0 = np.zeros(n)
        p0 = np.full(n, 0.5)
        for m in range(self.n_estimators):
            residuals = y - p0
            tree = DecisionTreeRegressor(max_depth=1, max_leaf_nodes=2)
            tree.fit(X, residuals)
            self.trees.append(tree)
            ids = tree.apply(X)
            for j in np.unique(ids):
                filter = (ids == j)
                num = residuals[filter].sum()
                den = (p0[filter] * (1 - p0[filter])).sum()
                if den == 0:
                    continue
                gamma = num / den
                f0[filter] += gamma * self.learning_rate
            p = sigmoid(f0)
            p0 = p

    def predict_proba(self, X):
        f = np.zeros(X.shape[0])
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        return sigmoid(f)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


data = load_iris()
X = data.data
y = data.target
le = LabelEncoder()
le.fit_transform(y)
new_X = X[y != 2]
new_y = y[y != 2]
X_train, X_test, y_train, y_test =train_test_split(new_X, new_y, test_size=0.2)
model1 = GBCFromScratch(learning_rate=0.1, n_estimators=10)
model1.fit(X_train, y_train)
preds1 = model1.predict(X_test)
print('Accuracy Score: ', accuracy_score(y_test, preds1))
model2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10)
model2.fit(X_train, y_train)
preds2 = model2.predict(X_test)
print('Accuracy Score: ', accuracy_score(y_test, preds2))


