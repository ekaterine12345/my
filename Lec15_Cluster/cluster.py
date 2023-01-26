import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
data = pd.read_csv('https://raw.githubusercontent.com/milaan9/Clustering-Datasets/master/01.%20UCI/wineqr.txt', sep=";")
data = data[["fixed acidity", "alcohol"]]
mycluster = KMeans(n_clusters=6, max_iter=300)
mycluster.fit(data)
labels = mycluster.predict(data)
print(silhouette_score(data, labels)) # ramdenad kargad daklasterda magas achvenebs
plt.scatter(data["fixed acidity"], data["alcohol"], c=labels)
plt.scatter(mycluster.cluster_centers_[:,0], mycluster.cluster_centers_[:, 1], s=70, c='green')
# titoeul clusters adebs mwvane wertils
plt.show()