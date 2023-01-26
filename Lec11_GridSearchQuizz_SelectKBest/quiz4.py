import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv('https://raw.githubusercontent.com/Athpr123/Binary-Classification-Using-Machine-learning/master/dataset.csv')
data.dropna(axis=0, inplace=True)
# print(data)
myLabel = LabelEncoder()
data['Agency'] = myLabel.fit_transform(data['Agency'])
data['Agency Type'] = myLabel.fit_transform(data['Agency Type'])
data['Destination'] = myLabel.fit_transform(data['Destination'])
data['Distribution Channel'] = myLabel.fit_transform(data['Distribution Channel'])
data['Gender'] = myLabel.fit_transform(data['Gender'])
data['Product Name'] = myLabel.fit_transform(data['Product Name'])
#print(data)

y = data['Claim'].values
X = data.drop('Claim', axis=1).values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.11, random_state=1)
model = AdaBoostClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05]}
hybrid = GridSearchCV(model, parameters, scoring='accuracy', cv=11)
hybrid.fit(X_train, y_train)
print(hybrid.score(X_train, y_train))
print(hybrid.score(X_test, y_test))

parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
              'n_estimators': [20, 25, 35]}
hybrid = GridSearchCV(model, parameters, scoring='accuracy', cv=11)
hybrid.fit(X_train, y_train)
print(hybrid.score(X_test, y_test))