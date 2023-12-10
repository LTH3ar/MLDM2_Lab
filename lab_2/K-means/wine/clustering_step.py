import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the iris dataset
iris = load_wine()
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

# Define the cluster colors and corresponding notes for wine dataset and centroids
cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}
cluster_notes = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 3: 'Centroids'}

# from 1 to 3 clusters show the plot, show name for each cluster color
for i in range(1, 4):
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)

    # Visualize the clusters
    pca = PCA(n_components=2)

    reduced_X = pd.DataFrame(data=pca.fit_transform(X), columns=['PCA1', 'PCA2'])

    # Reduced Features
    reduced_X.head()

    centers = pca.transform(kmeans.cluster_centers_)

    # reduced centers
    print(centers)

    plt.figure(figsize=(7, 5))
    for cluster, color in cluster_colors.items():
        plt.scatter(reduced_X['PCA1'][kmeans.labels_ == cluster],
                    reduced_X['PCA2'][kmeans.labels_ == cluster],
                    c=color, label=cluster_notes[cluster])

    # Scatter plot
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='red')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(f'Wine Cluster with Kmeans = {i}')
    plt.tight_layout()
    # show note for each cluster color
    plt.legend()

    plt.show()