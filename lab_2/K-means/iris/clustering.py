import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

# elbow method
wcss = {}
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss[i] = kmeans.inertia_

plt.plot(wcss.keys(), wcss.values(), 'gs-')
plt.xlabel("Values of 'k'")
plt.ylabel('WCSS')
plt.show()

# Apply k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(kmeans.cluster_centers_)

print(kmeans.labels_)

# Visualize the clusters
pca = PCA(n_components=2)

reduced_X = pd.DataFrame(data=pca.fit_transform(X), columns=['PCA1', 'PCA2'])

# Reduced Features
reduced_X.head()

centers = pca.transform(kmeans.cluster_centers_)

# reduced centers
print(centers)

plt.figure(figsize=(7, 5))

# Scatter plot
plt.scatter(reduced_X['PCA1'], reduced_X['PCA2'], c=kmeans.labels_)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='red')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Wine Cluster')
plt.tight_layout()
plt.show()