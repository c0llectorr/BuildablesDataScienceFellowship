#importing DS Libraries
from sklearn.datasets import load_iris

#loading dataset as a frame.
iris = load_iris(as_frame=True)

#Converting to DataFrame
df = iris.frame

#Calculating Central Tendencies of Sepal Lenghts:
mean = df['sepal length (cm)'].mean()
mode = df['sepal length (cm)'].mode()[0]
median = df['sepal length (cm)'].median()

#Printing Results
print(f"Central Tendencies of Sepal Lengths:\nMean: {mean}\nMedian: {median}\nMode: {mode}")
