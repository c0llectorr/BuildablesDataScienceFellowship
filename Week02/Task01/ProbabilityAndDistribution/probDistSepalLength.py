from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame

print(df.var())



