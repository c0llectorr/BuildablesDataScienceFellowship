from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["sepal length (cm)"]])

print("First 10 Min-Max scaled values for Sepal Length:")
print(scaled[:10].flatten())
