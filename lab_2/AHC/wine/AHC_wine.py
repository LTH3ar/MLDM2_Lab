import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the Iris dataset
from sklearn.datasets import load_wine
wine = load_wine()
wine_features = pd.DataFrame(wine.data, columns=wine.feature_names)

# AHC using Complete Linkage
hc_complete = linkage(wine_features, method='complete')

# Assign cluster labels based on a distance threshold (adjust as needed)
distance_threshold = 600  # You can adjust this value based on the dendrogram
cluster_labels = fcluster(hc_complete, t=distance_threshold, criterion='distance')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(hc_complete, labels=wine.target_names[wine.target], orientation='top',
           distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram - Wine Dataset (Complete Linkage)')
plt.xlabel('Class')
plt.ylabel('Distance')
plt.show()

# Display cluster count and size
cluster_count = len(np.unique(cluster_labels))
cluster_size = np.bincount(cluster_labels)

print(f"Number of clusters: {cluster_count}")
print(f"Cluster sizes: {cluster_size[1:]}")
# some info of the dataset
print(f"Number of samples: {wine_features.shape[0]}")
print(f"Number of features: {wine_features.shape[1]}")

exit()
