import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

# Use PCA to visualize the data distribution in 2D/3D with actual classes.

# Load the wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

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

# apply k-means clustering to pca
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# before pca
# calculate davies bouldin score
print('davies bouldin score: ', davies_bouldin_score(X, kmeans.labels_))
print("number of clusters: ", kmeans.n_clusters)
cluster_0 = 0
cluster_1 = 0
cluster_2 = 0
cluster_3 = 0
cluster_4 = 0
cluster_5 = 0
cluster_6 = 0
for i in kmeans.labels_:
    if i == 0:
        cluster_0 += 1
    elif i == 1:
        cluster_1 += 1
    elif i == 2:
        cluster_2 += 1
    elif i == 3:
        cluster_3 += 1
    elif i == 4:
        cluster_4 += 1
    elif i == 5:
        cluster_5 += 1
    else:
        cluster_6 += 1
print("number of samples for each cluster: ", cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6)

# Visualize the clusters
pca = PCA(n_components=2)

reduced_X = pd.DataFrame(data=pca.fit_transform(X), columns=['PCA1', 'PCA2'])

# Reduced Features
reduced_X.head()

centers = pca.transform(kmeans.cluster_centers_)
print(centers)

# Visualize the clusters without class
plt.figure(figsize=(8, 6))
for cluster in range(kmeans.n_clusters):
    plt.scatter(reduced_X['PCA1'][kmeans.labels_ == cluster], reduced_X['PCA2'][kmeans.labels_ == cluster])

plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=100, c='black', label='Centroids')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title(f'Wine Cluster with Kmeans = 2')
plt.legend()
plt.show()



# calculate davies bouldin score
print('davies bouldin score: ', davies_bouldin_score(X, kmeans.labels_))
print("number of clusters: ", kmeans.n_clusters)
cluster_0 = 0
cluster_1 = 0
cluster_2 = 0
cluster_3 = 0
cluster_4 = 0
cluster_5 = 0
cluster_6 = 0
for i in kmeans.labels_:
    if i == 0:
        cluster_0 += 1
    elif i == 1:
        cluster_1 += 1
    elif i == 2:
        cluster_2 += 1
    elif i == 3:
        cluster_3 += 1
    elif i == 4:
        cluster_4 += 1
    elif i == 5:
        cluster_5 += 1
    else:
        cluster_6 += 1
print("number of samples for each cluster: ", cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6)

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
for cluster in range(kmeans.n_clusters):
    ax.scatter3D(reduced_X['PCA1'][kmeans.labels_ == cluster], reduced_X['PCA2'][kmeans.labels_ == cluster], reduced_X['PCA3'][kmeans.labels_ == cluster])

ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', s=100, c='black', label='Centroids')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.legend()
plt.show()

# calculate davies bouldin score
print('davies bouldin score: ', davies_bouldin_score(X, kmeans.labels_))
cluster_0 = 0
cluster_1 = 0
print("number of clusters: ", kmeans.n_clusters)
cluster_0 = 0
cluster_1 = 0
cluster_2 = 0
cluster_3 = 0
cluster_4 = 0
cluster_5 = 0
cluster_6 = 0
for i in kmeans.labels_:
    if i == 0:
        cluster_0 += 1
    elif i == 1:
        cluster_1 += 1
    elif i == 2:
        cluster_2 += 1
    elif i == 3:
        cluster_3 += 1
    elif i == 4:
        cluster_4 += 1
    elif i == 5:
        cluster_5 += 1
    else:
        cluster_6 += 1
print("number of samples for each cluster: ", cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6)