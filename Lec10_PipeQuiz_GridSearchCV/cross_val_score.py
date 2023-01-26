from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, LeaveOneOut, ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
kFold = KFold(n_splits=3, shuffle=True, random_state=None)
stratifiedKFold = StratifiedKFold(n_splits=5, random_state=None)
loo = LeaveOneOut()
shuffleSplit = ShuffleSplit(n_splits=10, test_size=0.5, train_size=0.5 )
model = LogisticRegression(max_iter=5000)
scores = cross_val_score(model, iris.data, iris.target, cv=stratifiedKFold)
print(scores)
print(scores.mean())
# print(len(scores))
