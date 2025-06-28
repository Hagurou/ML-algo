import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(action='ignore')


def __init__(X, iters):
    n = X.shape[0]
    sample_weights = np.zeros((iters, n))
    stumps = np.zeros(iters, dtype=object)
    stump_weights = np.zeros(iters)
    errors = np.zeros(iters)
    return sample_weights, stumps, stump_weights, errors


def check(y):
    assert set(y) == {-1, 1}
    return y


def adaboost(X, y, iters=10):
    n = X.shape[0]
    y = check(y)
    sample_weights, stumps, stumps_weights, errors = __init__(X, iters)
    sample_weights[0] = np.ones(n) / n
    for i in range(iters):
        current_sew = sample_weights[i]
        stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        stump.fit(X, y, sample_weight=current_sew)

        stump_pred = stump.predict(X)
        error = current_sew[stump_pred != y].sum()
        error = max(error, 1e-10)
        stump_weight = np.log((1 - error) / error) / 2

        new_sew = current_sew * np.exp(-1 * stump_weight * stump_pred * y)

        new_sew = new_sew / np.sum(new_sew)

        if (i + 1) < iters:
            sample_weights[i + 1] = new_sew

        errors[i] = error
        stumps[i] = stump
        stumps_weights[i] = stump_weight
    return errors, stumps, stumps_weights


def predict(X, stumps, stump_weights):
    stump_preds = np.array([stump.predict(X) for stump in stumps])
    return np.sign(np.dot(stump_weights, stump_preds))



iris_data = load_iris()
X = iris_data.data
y = iris_data.target
le = LabelEncoder()
le.fit_transform(y)
new_X = X[y != 2]
new_y = y[y != 2]
new_y = np.where(new_y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2)
model1 = AdaBoostClassifier(n_estimators=50, random_state=42)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))
errors, stumps, stumps_weights = adaboost(X_train, y_train, iters=10)
model2 = predict(X_test, stumps, stumps_weights)
print("Adaboost from scratch model accuracy:", accuracy_score(y_test, model2))
