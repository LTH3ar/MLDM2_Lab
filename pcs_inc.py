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

# Define the range of components to consider
num_components_range = range(1, len(iris.feature_names) + 1)

# Calculate cumulative explained variance for each number of components
cumulative_explained_variance = []
for num_components in num_components_range:
    pca = PCA(n_components=num_components)
    data_pca = pca.fit_transform(data_scaled)
    cumulative_explained_variance.append(sum(pca.explained_variance_ratio_))

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(num_components_range, cumulative_explained_variance, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.grid(True)
plt.show()
