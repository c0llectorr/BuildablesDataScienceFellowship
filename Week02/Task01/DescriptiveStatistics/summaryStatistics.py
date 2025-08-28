#importing iris dataset
from sklearn.datasets import load_iris

#loading dataset and converting to dataFrame.
iris = load_iris(as_frame=True)
df = iris.frame

#summary statistics of dataset
print(f"Summary Statistics:")
def summary_statistics():
    mean = df.mean()
    median = df.median()
    variance = df.var()
    std = df.std()
    print(f"Mean:\n{mean}\n\nMedian:\n{median}\n\nVariance:\n{variance}\n\nStandard Variation:\n{std}\n")

summary_statistics()