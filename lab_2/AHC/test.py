import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Given dataset
X = np.array([1, 2, 9, 12, 20]).reshape(-1, 1)

# ndarray of labels
labels = np.array(['1', '2', '9', '12', '20'])

# AHC
# complete linkage
Z = linkage(X, method='complete')

# plot the AHC with labels
plt.figure(figsize=(10, 5))
plt.title('Complete Linkage with Labels')
dendrogram(Z, labels=labels)
plt.show()

# single linkage
Z = linkage(X, method='single')

# plot the AHC with labels
plt.figure(figsize=(10, 5))
plt.title('Single Linkage with Labels')
dendrogram(Z, labels=labels)
plt.show()