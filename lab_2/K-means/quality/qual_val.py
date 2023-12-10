import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn import datasets

# Load Wine dataset
wine = datasets.load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Load Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# davies_bouldin_score
# Davies-Bouldin score for Iris dataset
davies_bouldin_score_iris = []
for i in range(2, 11):
    kmeans_iris = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans_iris.fit(iris_df)
    davies_bouldin_score_iris.append(davies_bouldin_score(iris_df, kmeans_iris.labels_))

# Davies-Bouldin score for Wine dataset
davies_bouldin_score_wine = []
for i in range(2, 11):
    kmeans_wine = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans_wine.fit(wine_df)
    davies_bouldin_score_wine.append(davies_bouldin_score(wine_df, kmeans_wine.labels_))

# Plot Davies-Bouldin score add dot for each k with grid
# main plot to cover all subplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), davies_bouldin_score_iris, marker='o')
plt.title('Davies-Bouldin score for Iris Dataset')
plt.xlabel('Number of clusters')
plt.ylabel('Davies-Bouldin score')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), davies_bouldin_score_wine, marker='o')
plt.title('Davies-Bouldin score for Wine Dataset')
plt.xlabel('Number of clusters')
plt.ylabel('Davies-Bouldin score')
plt.grid()
plt.show()