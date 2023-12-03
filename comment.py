import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Calculate correlation matrix
correlation_matrix = data.corr()

# Find most correlated features
most_correlated = correlation_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()

# Display the most correlated features
print("Most Correlated Features:")
print(most_correlated)
