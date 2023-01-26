import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('https://raw.githubusercontent.com/Mashimo/datascience/master/datasets/wheat.data')
# print(data.isnull().any())
data['compactness'] = data['compactness'].fillna(data['compactness'].mean())
print(data.corr())
data.dropna(axis=0, inplace=True)

myLabel = LabelEncoder()
data['wheat_type'] = myLabel.fit_transform(data['wheat_type'])
data.drop("id", axis=1, inplace=True)
print(data)

y = data["wheat_type"].values
X = data.drop("wheat_type", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
my_algo = KNeighborsClassifier(n_neighbors=5)
my_algo.fit(X_train, y_train)
print(my_algo.score(X_test, y_test))
