import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
x = np.arange(10).reshape(-1, 1)
print(x)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
print(y)
# create model
model = LogisticRegression(solver='liblinear', random_state=0)
# print(model.classes_)
model.fit(x, y)
print(model.classes_)
print("Model Intercept is {}".format(model.intercept_))
print("model  coefficient is {}".format(model.coef_))
print(model.predict_proba(x))
print(model.predict(x))
print(model.score(x, y))
print(confusion_matrix(y, model.predict(x)))
print(classification_report(y,model.predict(x)))