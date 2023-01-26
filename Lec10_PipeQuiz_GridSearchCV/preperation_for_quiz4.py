import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=40, n_informative=9, n_redundant=31, n_classes=3, random_state=1)
model = SVC()
scores = cross_val_score(model, X, y, scoring='accuracy', cv=30, n_jobs=-1)
print(scores)
print(np.mean(scores))
parameters = {'degree': [1, 2, 3, 4, 5, 6, 7, 8]}
hybrid = GridSearchCV(model, parameters, scoring='accuracy', cv=3, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_)