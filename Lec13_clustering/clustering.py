import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/JangirSumit/kmeans-clustering/master/driver-data.csv')
data.drop("id", axis=1, inplace=True) # id სვეტი გადავაგდე
print(data.head())
# print(data.info())
from sklearn.metrics import silhouette_score
score = {}
for i in range(2, 5):
    mycluster = KMeans(n_clusters=i, max_iter=4000)
    mycluster.fit(data)
    y_predicted = mycluster.predict(data)
    print("y = ", y_predicted)
    print(silhouette_score(data, y_predicted))
    score[i] = silhouette_score(data, y_predicted)
print(score)
plt.scatter(data['mean_dist_day'], data['mean_over_speed_perc'], c=y_predicted)
plt.scatter(mycluster.cluster_centers_[:,0], mycluster.cluster_centers_[:,1], s=100, c="pink")
plt.show()


