import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

Data = pd.read_csv("https://raw.githubusercontent.com/Mashimo/datascience/master/datasets/wheat.data")
print(Data.isnull().any())
# print(Data.head())
print("*"*100)
Data.dropna(axis=0, inplace=True)  # Drop rows which contain missing values.
print(Data.head())
my_label = LabelEncoder()  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# print("lab ", type(my_label))
Data["wheat_type"] = my_label.fit_transform(Data["wheat_type"])

print("%"*100)
Data.drop("id", axis=1, inplace=True)
# print(Data)
print("%"*100)

y = Data["wheat_type"].values
x = Data.drop("wheat_type", axis=1).values
print(x)
print("y = ", y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

print("x_train ", X_train)
print("X_test ", X_test)
print("y_train ", y_train)
print("y_test ", y_test)


my_algo = KNeighborsClassifier()
print("my_algo", my_algo)
my_algo.fit(X_train, y_train)
print(my_algo.score(X_test, y_test)) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
new_x = np.array([[15.27, 14.85, 8.8715, 5.765, 3.317, 2.225, 5.225]])
print(my_algo.predict(new_x))
