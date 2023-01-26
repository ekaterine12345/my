import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2
data = pd.read_csv('credit.csv')
print(data)
y = data["y"].values
X = data.drop("y", axis=1).values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

selector = SelectKBest(score_func=f_classif, k=6)
X_selected = selector.fit_transform(X, y)
print(selector.get_feature_names_out())
print(X_selected)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=1)
model1 = SVC()
model1.fit(X_train, y_train)
print(model1.score(X_test, y_test))
