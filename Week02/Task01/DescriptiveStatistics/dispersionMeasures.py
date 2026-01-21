#importing iris dataset.
from sklearn.datasets import load_iris

#loading data and converting into dataFrame.
iris = load_iris(as_frame=True)
df = iris.frame

#Calculating Variance and Standard Deviation.
def calculateVarianceAndSTD(columnName):
    variance = df[columnName].var()
    std = df[columnName].std()
    return variance, std

#main function
def main ():
    #As we have to calculate petals width.
    columnName = 'petal width (cm)'
    variance, std = calculateVarianceAndSTD(columnName)
    print(f"Dispersion Measures:\nVariance: {variance}\nStandard Deviation: {std}")
    
if __name__ == "__main__":
    main()