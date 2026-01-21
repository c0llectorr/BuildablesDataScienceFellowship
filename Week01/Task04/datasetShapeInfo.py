import sklearn.datasets as datasets
import pandas as pd

def datasetShapeInfo(dataset_name):
    avaiable_datasets = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer
    }
    if dataset_name not in avaiable_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(avaiable_datasets.keys())}.")
   
    dataset = avaiable_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    return df.shape

def main():
    dataset_name = input("Enter dataset name: ")
    print(datasetShapeInfo(dataset_name))
    
if __name__ == "__main__":
    main()