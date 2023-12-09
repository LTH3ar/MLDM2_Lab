import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Calculate mean, variance, covariance, correlation
mean_values = data.mean()
variance_values = data.var()
covariance_matrix = data.cov()
correlation_matrix = data.corr()

# Display in table format
statistics_table = pd.DataFrame({
    'Mean': mean_values,
    'Variance': variance_values,
})

# Display the table
print("Mean and Variance:")
print(statistics_table)

# Display covariance and correlation matrices
print("\nCovariance Matrix:")
print(covariance_matrix)

print("\nCorrelation Matrix:")
print(correlation_matrix)

# save the table to a csv file
statistics_table.to_csv("iris_statistics.csv", index=True)
covariance_matrix.to_csv("iris_covariance.csv", index=True)
correlation_matrix.to_csv("iris_correlation.csv", index=True)