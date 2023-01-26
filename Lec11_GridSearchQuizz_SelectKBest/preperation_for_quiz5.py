import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile
from sklearn.naive_bayes import GaussianNB
heart = pd.read_csv("https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv")
print(heart.head())
y = heart['target']
X = heart.drop('target', axis=1)
selector = SelectPercentile(score_func=f_classif, percentile=30)
X_selected = selector.fit_transform(X, y)
print(selector.get_feature_names_out())
print(X_selected)

pipe = Pipeline([('percentile', SelectPercentile()),
                 ('gaus', GaussianNB())])
parameter_list = {'percentile__percentile': [20, 40, 60, 80, 90, 95]}
hybrid = GridSearchCV(pipe, parameter_list, scoring='accuracy', cv=3, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_)
print(hybrid.best_score_)
