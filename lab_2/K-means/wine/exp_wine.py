import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets

# Load Wine dataset
wine = datasets.load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Elbow Method
wcss_wine = []
for i in range(1, 11):
    kmeans_wine = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_wine.fit(wine_df)
    wcss_wine.append(kmeans_wine.inertia_)

# Plot Elbow Method
plt.plot(range(1, 11), wcss_wine)
plt.title('Elbow Method for Wine Dataset')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

import itertools
# Choose k based on the elbow method (let's say k=3)

# KMeans for Wine Dataset
kmeans_wine = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
wine_df['Cluster'] = kmeans_wine.fit_predict(wine_df)
# Get all possible combinations of attributes
attribute_combinations = list(itertools.combinations(range(len(wine.feature_names)), 2))
centroids = kmeans_wine.cluster_centers_

# Define the cluster colors and corresponding notes for wine dataset and centroids
cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}
cluster_notes = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 3: 'Centroids'}

# Plot scatterplots for all combinations
for attr1, attr2 in attribute_combinations:
    plt.scatter(wine_df.iloc[:, attr1], wine_df.iloc[:, attr2], c=wine_df['Cluster'], cmap='viridis')
    '''
    plt.figure(figsize=(7, 5))
    for cluster, color in cluster_colors.items():
        plt.scatter(wine_df.iloc[:, attr1][wine_df['Cluster'] == cluster],
                    wine_df.iloc[:, attr2][wine_df['Cluster'] == cluster],
                    c=color, label=cluster_notes[cluster])
    '''
    plt.scatter(centroids[:, attr1], centroids[:, attr2], marker='*', s=200, linewidths=1, color='black',
                label='Centroids')
    plt.title(f'KMeans Clustering for Wine Dataset ({wine.feature_names[attr1]} vs {wine.feature_names[attr2]})')
    plt.xlabel(wine.feature_names[attr1])
    plt.ylabel(wine.feature_names[attr2])
    plt.legend()
    plt.show()