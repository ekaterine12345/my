import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

myLabel = LabelEncoder()
data['sex'] = myLabel.fit_transform(data['sex'])
data['smoker'] = myLabel.fit_transform(data['smoker'])
data['region'] = myLabel.fit_transform(data['region'])

y = data['charges'].values
X = data.drop('charges', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

myLinear = LinearRegression()
myLinear.fit(X_train, y_train)
print(myLinear.score(X_test, y_test))

hybrid = Pipeline([("scalar", MinMaxScaler(feature_range=(0, 1))),
                    ("PCA", PCA(n_components=5)),
                    ("Algorithm", LinearRegression())])
hybrid.fit(X_train, y_train)
print(hybrid.score(X_test, y_test))
# 24 noemberi; quizi 3


