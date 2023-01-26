import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import dendrogram, linkage
data = pd.read_csv('https://raw.githubusercontent.com/JangirSumit/kmeans-clustering/master/driver-data.csv')
print(data)
data.drop("id", axis=1, inplace=True)

scores = {}
for i in range(2, 5):
    mycluster = AgglomerativeClustering(n_clusters=i, linkage='single')
    # mycluster.fit(data)
    y_predicted = mycluster.fit_predict(data)
    scores[i] = silhouette_score(data, y_predicted)
    print(scores[i])

plt.scatter(data["mean_dist_day"], data["mean_over_speed_perc"], c=y_predicted)
plt.figure(figsize=(10,7))

# from scipy.cluster.hierarchy import dendrogram, linkage
# linked =linkage(X, 'single')
# dendrogram(linked,orientation='top',labels=labels,distance_sort='descending',show_leaf_counts=True)

# plt.scatter(mycluster.cluster_centers_[:, 0], mycluster.cluster_centers_[:, 1], s=70, c="green")
plt.show()