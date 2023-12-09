import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
iris_features = pd.DataFrame(iris.data, columns=iris.feature_names)

# AHC using Complete Linkage
hc_complete = linkage(iris_features, method='single')

# Assign cluster labels based on a distance threshold (adjust as needed)
distance_threshold = 0.8  # You can adjust this value based on the dendrogram
cluster_labels = fcluster(hc_complete, t=distance_threshold, criterion='distance')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(hc_complete, labels=iris.target_names[iris.target], orientation='top',
           distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram - Iris Dataset (Single Linkage)')
plt.xlabel('Species')
plt.ylabel('Distance')
plt.show()

# Display cluster count and size
cluster_count = len(np.unique(cluster_labels))
cluster_size = np.bincount(cluster_labels)

print(f"Number of clusters: {cluster_count}")
print(f"Cluster sizes: {cluster_size[1:]}")
exit()
