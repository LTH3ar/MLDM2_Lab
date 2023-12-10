import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}
cluster_notes = {0: 'Sentosa', 1: 'Versicolor', 2: 'Virginica', 3: 'Centroids'}

# Use PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

# Visualize the data distribution with actual classes
plt.figure(figsize=(8, 6))

for i in range(len(cluster_colors)):
    plt.scatter(X_2D[df.index[iris.target == i], 0], X_2D[df.index[iris.target == i], 1], label=f'Class {i}', c=cluster_colors[i], alpha=0.7)

plt.title('PCA Visualization of Iris Dataset (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Use PCA to reduce dimensionality to 3 components
pca = PCA(n_components=3)
X_3D = pca.fit_transform(X)

# Visualize the data distribution with actual classes
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')

for i in range(len(cluster_colors)):
    ax.scatter3D(X_3D[df.index[iris.target == i], 0], X_3D[df.index[iris.target == i], 1], X_3D[df.index[iris.target == i], 2], label=f'Class {i}', c=cluster_colors[i], alpha=0.7)

ax.set_title('PCA Visualization of Iris Dataset (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.legend()
plt.show()
