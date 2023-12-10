import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

# Use PCA to visualize the data distribution in 2D/3D with actual classes.

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Check the data information
df.info()

scaler = StandardScaler()

features = scaler.fit(df)
features = features.transform(df)

# Convert to pandas Dataframe
scaled_df = pd.DataFrame(features, columns=df.columns)
# Print the scaled data
scaled_df.head(2)

X = scaled_df.values

# Define the cluster colors and corresponding notes
cluster_colors = {0: 'red', 1: 'green'}
cluster_notes = {0: 'class 0', 1: 'class 1'}

# apply k-means clustering to pca
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# before pca
plt.figure(figsize=(8, 6))
for cluster, color in cluster_colors.items():
    plt.scatter(X[kmeans.labels_ == cluster, 0],
                X[kmeans.labels_ == cluster, 1],
                c=color, label=cluster_notes[cluster])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=100, c='black', label='Centroids')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title(f'Iris Cluster with Kmeans = 2')
plt.legend()
plt.show()
# calculate davies bouldin score
print('davies bouldin score: ', davies_bouldin_score(X, kmeans.labels_))
print("number of clusters: ", kmeans.n_clusters)
cluster_0 = 0
cluster_1 = 0
for i in kmeans.labels_:
    if i == 0:
        cluster_0 += 1
    else:
        cluster_1 += 1
print("number of samples for each cluster: ", cluster_0, cluster_1)

# Visualize the clusters
pca = PCA(n_components=2)

reduced_X = pd.DataFrame(data=pca.fit_transform(X), columns=['PCA1', 'PCA2'])

# Reduced Features
reduced_X.head()

centers = pca.transform(kmeans.cluster_centers_)
print(centers)

# Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster, color in cluster_colors.items():
    plt.scatter(reduced_X['PCA1'][kmeans.labels_ == cluster],
                reduced_X['PCA2'][kmeans.labels_ == cluster],
                c=color, label=cluster_notes[cluster])

plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=100, c='black', label='Centroids')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title(f'Iris Cluster with Kmeans = 2')
plt.legend()
plt.show()
# calculate davies bouldin score
print('davies bouldin score: ', davies_bouldin_score(X, kmeans.labels_))
print("number of clusters: ", kmeans.n_clusters)
cluster_0 = 0
cluster_1 = 0
for i in kmeans.labels_:
    if i == 0:
        cluster_0 += 1
    else:
        cluster_1 += 1
print("number of samples for each cluster: ", cluster_0, cluster_1)

# kmeans with pca 3d
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Visualize the clusters
pca = PCA(n_components=3)

reduced_X = pd.DataFrame(data=pca.fit_transform(X), columns=['PCA1', 'PCA2', 'PCA3'])

# Reduced Features
reduced_X.head()

centers = pca.transform(kmeans.cluster_centers_)

# Visualize the clusters
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
for cluster, color in cluster_colors.items():
    ax.scatter3D(reduced_X['PCA1'][kmeans.labels_ == cluster],
                 reduced_X['PCA2'][kmeans.labels_ == cluster],
                 reduced_X['PCA3'][kmeans.labels_ == cluster],
                 c=color, label=cluster_notes[cluster])

ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', s=100, c='black', label='Centroids')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.title(f'Iris Cluster with Kmeans = 2')
plt.legend()
plt.show()
# calculate davies bouldin score
print('davies bouldin score: ', davies_bouldin_score(X, kmeans.labels_))
cluster_0 = 0
cluster_1 = 0
print("number of clusters: ", kmeans.n_clusters)
for i in kmeans.labels_:
    if i == 0:
        cluster_0 += 1
    else:
        cluster_1 += 1
print("number of samples for each cluster: ", cluster_0, cluster_1)