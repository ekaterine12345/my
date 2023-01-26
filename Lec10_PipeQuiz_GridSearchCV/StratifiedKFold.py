from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
accuracy = []
logreg = LogisticRegression()
skf = StratifiedKFold(n_splits=5, random_state=None)
# for train_index,test_index in skf.split(X,Y):
#     X_train, X_test= X[train_index],X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#     logreg.fit(X_train,Y_train)
#     prediction = logreg.predict(X_test)
#     score = accuracy_score(prediction,Y_test)
#     accuracy.append(score)
# print(accuracy)
