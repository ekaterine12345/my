import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
data = pd.read_csv('https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv')
X = data.drop('target', axis=1).values
print(X)
y = data['target'].values
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
algo1 = LogisticRegression(max_iter=500000, C=0.7)
algo1.fit(X_train, y_train)
print(algo1.score(X_test, y_test))

algo2 = GaussianNB(priors=[0.3, 0.7])
algo2.fit(X_train, y_train)
print(algo2.score(X_test, y_test))

algo3 = MultinomialNB()
algo3.fit(X_train, y_train)
print(algo3.score(X_test, y_test))

algo4 = BernoulliNB()
algo4.fit(X_train, y_train)
print(algo4.score(X_test, y_test))
