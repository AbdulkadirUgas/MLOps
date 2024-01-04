import numpy as nb
from sklearn.datasets import load_iris

#load the IRIS dataset
data = load_iris()

#print the features 
print("Features name:",data.feature_names)
# Print the target names
print("Target names:", data.target_names)

# print first 5 samples
print("first 5 sample")
print(data.data[:5])
print(data.target[:5])