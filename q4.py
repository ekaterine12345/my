import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv('https://raw.githubusercontent.com/Athpr123/Binary-Classification-Using-Machine-learning/master/dataset.csv')
print(data.head())
data.dropna(axis=0, inplace=True)
label = LabelEncoder()
data['Agency'] = label.fit_transform(data['Agency'])
data['Agency Type'] = label.fit_transform(data['Agency Type'])
data['Destination'] = label.fit_transform(data['Destination'])
data['Distribution Channel'] = label.fit_transform(data['Distribution Channel'])
data['Gender'] = label.fit_transform(data['Gender'])
data['Product Name'] = label.fit_transform(data['Product Name'])

y = data["Claim"].values
X = data.drop("Claim", axis=1)

print(X, y)
model = AdaBoostClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.11)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05]}
hybrid = GridSearchCV(model, parameters, scoring='accuracy', cv=3, n_jobs=-1)
hybrid.fit(X_train, y_train)
print(hybrid.best_score_)

parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05], 'n_estimators': [20, 25, 35]}
hybrid2 = GridSearchCV(model, parameters, scoring='accuracy', cv=3, n_jobs=-1)
hybrid2.fit(X_train, y_train)
print(hybrid2.best_score_)
print(hybrid2.best_params_)

