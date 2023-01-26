import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


irisi = load_iris()
print(irisi)
# print("~"*100)
# print(irisi.data)
# print("*"*100)
print(irisi.feature_names , irisi.data.shape)
print('+'*100)
#print(irisi.target)
# print(irisi.target_names, irisi.target.shape)
X = irisi.data
y = irisi.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# print(X_train.shape)
# print(X_test.shape)
K_range = range(1, 26)
print(K_range.stop)
scores = {}
scores_list = []
for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # print(knn)
    # print(knn.n_neighbors)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    # print(k, y_test, y_pred, scores[k])
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(K_range, scores_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing Accuracy')
# plt.show()