import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Display the table
print(data)

# save the table to a csv file
data.to_csv("iris.csv", index=True)
