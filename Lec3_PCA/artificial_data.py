import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=2000, n_features=15, n_informative=2, n_redundant=10, n_classes=3, n_repeated=3,
                           n_clusters_per_class=1, flip_y=0.3, random_state=1)
# 15 სვეტიდან 2 სვეტი არის მნიშვნელობანი 13 ზედმეტი. 2000 სტრიქნოია სულ.
# n_classes - სულ რამდენი კლასია
# n_repeated -
print(X.shape)
print(y)
data = pd.DataFrame(X, columns=["X"+str(i) for i in range(15)])
data["target"] = y
print(data.corr())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
my_algo = KNeighborsClassifier()
my_algo.fit(X_train, y_train)
print(my_algo.score(X_test, y_test))
# x ebs shoris rogoria kavshiri. da x ebs da y s shoris

from sklearn.decomposition import PCA, KernelPCA
#from sklearn.manifold
transform = PCA(n_components=2)  # 15dan sheiqmneba 2 akali sveti
X_train = transform.fit_transform(X_train) #  აქ ნასწავლით
X_test = transform.transform(X_test)  # ეს უნდა დაფორმატირდეს
print(np.sum(transform.explained_variance_ratio_))
my_algo.fit(X_train, y_train)
print(my_algo.score(X_test, y_test))