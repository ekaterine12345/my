import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
heart = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv")
print(heart.head())
y = heart['target'].values
# X =heart.drop('target',axis=1).values
X = heart[["age", "sex", "cp", "fbs", "ca", "exang"]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
algo1 = LogisticRegression(max_iter=500000, C=0.7)
algo1.fit(X_train, y_train)
print(algo1.score(X_test, y_test))
algo2 = GaussianNB(priors=[0.3, 0.7])
algo3 = MultinomialNB()
algo4 = BernoulliNB()
algo4.fit(X_train, y_train)
print(algo4.score(X_test, y_test))
# me6 leeciaze dawerili kodi