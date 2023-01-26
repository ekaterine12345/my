import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC

X, y = make_classification(n_samples=10000, n_features=40, n_informative=9, n_redundant=31, n_classes=3, random_state=1)
model = SVC(kernel='poly')

# scores = cross_val_score(model, X, y, scoring='accuracy', cv=30, n_jobs=-1) # neg_mean
# print(np.mean(scores))
parameters = {"degree": [1, 2, 3, 4, 5, 6, 7, 8]}

hybrid = GridSearchCV(model, parameters, scoring='accuracy', cv=3, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_) # {'degree': 3}


# quizze X da y datasetidan unda wamovigo da sheileba SVC-s magier sxva algorithmi ikos
# jvaredini varidaciis meshveobit gaiget algoritmis sahualo scori da amistvis monacemebi gakavit 30 mnawulad
# n_jobs=-1   - prodecsoris kvela birtvi datvirtos