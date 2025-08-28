from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame

sepal_length_probs = df['sepal length (cm)'].value_counts(normalize=True).sort_index()
print(sepal_length_probs)