from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame

grouped = df.groupby("target")["sepal length (cm)"].mean()

print("Mean Sepal Length for each species:")
print(grouped)
