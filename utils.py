from sklearn import datasets
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

def get_regresion_dataset():
    # Generate a synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    print("X ->", X.shape)
    print("y ->", y.shape)
    return X, y

def get_decision_tree_dataset():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    return X, y