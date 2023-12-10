import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def elbow_method_dual_plot(X_dataset_1, X_dataset_2):
    # plot the elbow method for both datasets side by side
    # elbow method

    # for dataset 1
    wcss = {}
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_dataset_1)
        wcss[i] = kmeans.inertia_

    plt.subplot(1, 2, 1)
    plt.plot(wcss.keys(), wcss.values(), 'gs-')
    plt.xlabel("Values of 'k' | Number of Clusters")
    plt.xticks(range(1, 11))
    plt.ylabel('Cluster Sum of Squares')
    plt.title('Elbow Method for Iris Dataset')

    # for dataset 2

    wcss = {}

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_dataset_2)
        wcss[i] = kmeans.inertia_

    plt.subplot(1, 2, 2)
    plt.plot(wcss.keys(), wcss.values(), 'gs-')
    plt.xlabel("Values of 'k' | Number of Clusters")
    plt.xticks(range(1, 11))
    plt.ylabel('Cluster Sum of Squares')
    plt.title('Elbow Method for Wine Dataset')
    # show the plot
    plt.show()


# Scaler
scaler = StandardScaler()

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# to csv
iris_df.to_csv('iris.csv', index=False)
iris_df.info()
features_iris = scaler.fit(iris_df)
features_iris = features_iris.transform(iris_df)
scaled_iris_df = pd.DataFrame(features_iris, columns=iris_df.columns)
scaled_iris_df.head(2)
X_iris = scaled_iris_df.to_numpy()

# Load the wine dataset
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
# to csv
wine_df.to_csv('wine.csv', index=False)
wine_df.info()
features_wine = scaler.fit(wine_df)
features_wine = features_wine.transform(wine_df)
scaled_wine_df = pd.DataFrame(features_wine, columns=wine_df.columns)
scaled_wine_df.head(2)
X_wine = scaled_wine_df.to_numpy()

# elbow method
elbow_method_dual_plot(X_iris, X_wine)

# Apply k-means clustering
kmeans_iris = KMeans(n_clusters=3)
kmeans_iris.fit(X_iris)

kmeans_wine = KMeans(n_clusters=3)
kmeans_wine.fit(X_wine)

# Visualize the clusters
pca_iris = PCA(n_components=2)
pca_wine = PCA(n_components=2)

reduced_X_iris = pd.DataFrame(data=pca_iris.fit_transform(X_iris), columns=['PCA1', 'PCA2'])
reduced_X_wine = pd.DataFrame(data=pca_wine.fit_transform(X_wine), columns=['PCA1', 'PCA2'])

# Reduced Features
reduced_X_iris.head()
reduced_X_wine.head()

centers_iris = pca_iris.transform(kmeans_iris.cluster_centers_)
centers_wine = pca_wine.transform(kmeans_wine.cluster_centers_)

# Define the cluster colors and corresponding notes for iris dataset and centroids
cluster_colors_iris = {0: 'red', 1: 'green', 2: 'blue'}
cluster_notes_iris = {0: 'Sentosa', 1: 'Versicolor', 2: 'Virginica'}

# Define the cluster colors and corresponding notes for wine dataset and centroids
cluster_colors_wine = {0: 'red', 1: 'green', 2: 'blue'}
cluster_notes_wine = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3'}

plt.figure(figsize=(14, 5))

# Scatter plot for iris dataset
plt.subplot(1, 2, 1)
for cluster, color in cluster_colors_iris.items():
    plt.scatter(reduced_X_iris['PCA1'][kmeans_iris.labels_ == cluster],
                reduced_X_iris['PCA2'][kmeans_iris.labels_ == cluster],
                c=color, label=cluster_notes_iris[cluster])

# Scatter plot for wine dataset
plt.subplot(1, 2, 2)
for cluster, color in cluster_colors_wine.items():
    plt.scatter(reduced_X_wine['PCA1'][kmeans_wine.labels_ == cluster],
                reduced_X_wine['PCA2'][kmeans_wine.labels_ == cluster],
                c=color, label=cluster_notes_wine[cluster])

# Scatter plot for centroids
plt.subplot(1, 2, 1)
plt.scatter(centers_iris[:, 0], centers_iris[:, 1], marker='x', s=100, c='red', label='Centroids')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Iris Cluster')
plt.tight_layout()
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(centers_wine[:, 0], centers_wine[:, 1], marker='x', s=100, c='red', label='Centroids')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Wine Cluster')
plt.tight_layout()
plt.legend()

plt.show()
print(kmeans_iris.labels_)
print(kmeans_wine.labels_)
exit()







