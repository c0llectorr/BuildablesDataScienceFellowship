from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame

X = df[["sepal length (cm)"]]
y = df["petal length (cm)"]

model = LinearRegression()
model.fit(X, y)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
