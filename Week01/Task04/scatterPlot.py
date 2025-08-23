from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
df = iris.frame

plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], alpha=0.7)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Scatterplot: Sepal Length vs Petal Length")
plt.show()
