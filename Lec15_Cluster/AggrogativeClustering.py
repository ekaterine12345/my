import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/JangirSumit/kmeans-clustering/master/driver-data.csv')
print(data)
data.drop("id", axis=1, inplace=True)

labels = range(1, 23)
X = np.array(data[["mean_dist_day", "mean_over_speed_perc"]].values)
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(X, 'single')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=True)
plt.show()
