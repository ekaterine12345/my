import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

data = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/DAAG/possum.csv').iloc[:, 1:]
print(data.head())

myLabel = LabelEncoder()
data['Pop'] = myLabel.fit_transform(data['Pop'])
data['sex'] = myLabel.fit_transform(data['sex'])
data.dropna(axis=0, inplace=True)
# print(data.head())
# print(data.isnull().sum)

y = data['footlgth'].values
X = data.drop('footlgth', axis=1).values
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

hybrid = Pipeline(steps=[("scale", StandardScaler()),
                         ("PCA", PCA(n_components=12)),
                          ("Algorithm", LinearRegression())])
hybrid.fit(X_train, y_train)
print(y_train)
print(hybrid.score(X_test, y_test))