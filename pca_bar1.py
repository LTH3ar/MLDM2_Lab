import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA with 2 components
pca = PCA(n_components=4)
data_pca = pca.fit_transform(data_scaled)

# Proportion of variance explained by each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Total variance explained by the first two principal components
total_variance_explained = sum(explained_variance_ratio)
loss = 1 - total_variance_explained

# Bar chart
components = [f"PC{i}" for i in range(1, len(explained_variance_ratio) + 1)]

plt.bar(components, explained_variance_ratio, color=['blue', 'orange'])
plt.xlabel('Principal Components')
plt.ylabel('Proportion of Variance Explained')
plt.title('Proportion of Variance Explained by Principal Components')
plt.show()

print(f"Proportion of Variance Explained by PC1: {explained_variance_ratio[0]}")
print(f"Proportion of Variance Explained by PC2: {explained_variance_ratio[1]}")
print(f"Proportion of Variance Explained by PC3: {explained_variance_ratio[2]}")
print(f"Proportion of Variance Explained by PC4: {explained_variance_ratio[3]}")
print(f"Total Variance Explained by PC1 and PC2: {total_variance_explained}")
print(f"Loss: {loss}")
