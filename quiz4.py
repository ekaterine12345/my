import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

data = pd.read_csv('https://raw.githubusercontent.com/Athpr123/Binary-Classification-Using-Machine-learning/master/dataset.csv')
print(data.head())
data.dropna(axis=0, inplace=True)
myLabel = LabelEncoder()
data['Agency'] = myLabel.fit_transform(data['Agency'])
data['Agency Type'] = myLabel.fit_transform(data['Agency Type'])
data['Destination'] = myLabel.fit_transform(data['Destination'])
data['Distribution Channel'] = myLabel.fit_transform(data['Distribution Channel'])
data['Gender'] = myLabel.fit_transform(data['Gender'])
data['Product Name'] = myLabel.fit_transform(data['Product Name'])

print(data.head())
X = data.drop('Claim', axis=1).values
y = data['Claim'].values
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.11)
model = AdaBoostClassifier()
score1 = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=9, n_jobs=-1)
score2 = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=9, n_jobs=-1)
print(score1)  # [0.73775601 0.74309884 0.73552983 0.73552983 0.73285841 0.72885129 0.74487979 0.74487979 0.74042743]
print(score2)  # [0.6942446  0.68705036 0.74820144 0.76618705 0.72661871 0.71223022 0.6967509  0.72202166 0.76173285]

parameters = {"learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05]}

hybrid = GridSearchCV(model, parameters, scoring='accuracy', cv=9, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_score_)

parameters = {"learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05], "n_estimators": [20, 25, 35]}
hybrid2 = GridSearchCV(model, parameters, scoring='accuracy', cv=9, n_jobs=-1)
hybrid2.fit(X, y)
print(hybrid2.best_score_)
print(hybrid2.best_params_)