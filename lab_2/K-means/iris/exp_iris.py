import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets

# Load Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(iris_df)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Iris Dataset')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

import itertools
# Choose k based on the elbow method (let's say k=3)

# KMeans for Iris Dataset
kmeans_iris = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
iris_df['Cluster'] = kmeans_iris.fit_predict(iris_df)

# Get all possible combinations of attributes
attribute_combinations = list(itertools.combinations(range(len(iris.feature_names)), 2))

# Define the cluster colors and corresponding notes for wine dataset and centroids
cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}
cluster_notes = {0: 'sentosa', 1: 'versicolor', 2: 'virginica'}

# Plot scatterplots for all combinations
for attr1, attr2 in attribute_combinations:
    plt.scatter(iris_df.iloc[:, attr1], iris_df.iloc[:, attr2], c=iris_df['Cluster'], cmap='viridis')
    '''
    plt.figure(figsize=(7, 5))
    for cluster, color in cluster_colors.items():
        plt.scatter(iris_df.iloc[:, attr1][iris_df['Cluster'] == cluster],
                    iris_df.iloc[:, attr2][iris_df['Cluster'] == cluster],
                    c=color, label=cluster_notes[cluster])
    '''
    plt.scatter(kmeans_iris.cluster_centers_[:, attr1], kmeans_iris.cluster_centers_[:, attr2], marker='*', s=200, linewidths=1, color='black', label='Centroids')
    plt.title(f'KMeans Clustering for Iris Dataset ({iris.feature_names[attr1]} vs {iris.feature_names[attr2]})')
    plt.xlabel(iris.feature_names[attr1])
    plt.ylabel(iris.feature_names[attr2])
    plt.legend()
    plt.show()