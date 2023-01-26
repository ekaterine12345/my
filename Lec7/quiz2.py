import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

data = pd.read_csv('https://raw.githubusercontent.com/doguilmak/Heart-Diseaseor-Attack-Classification/main/heart_disease_health_indicators_BRFSS2015.csv')
# print(data)
y = data["HeartDiseaseorAttack"].values
X = data.drop("HeartDiseaseorAttack", axis=1).values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

myLogistic = LogisticRegression(max_iter=50000, C=0.7)
myLogistic.fit(X_train, y_train)
print(myLogistic.score(X_test, y_test))

myGaussian = GaussianNB(priors=[0.3, 0.7])
myGaussian.fit(X_train, y_train)
print(myGaussian.score(X_test, y_test))

X1 = data[["Fruits", "PhysActivity", "HvyAlcoholConsump"]]
myBernoulli = BernoulliNB()

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=1)
myBernoulli.fit(X_train, y_train)
print(myBernoulli.score(X_test, y_test))


