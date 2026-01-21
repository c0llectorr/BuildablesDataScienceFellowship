import sklearn.datasets as datasets
import pandas as pd

def load_dataset(dataset_name):
    available_datasets = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(available_datasets.keys())}.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    return df
def main():
    dataset_name = input("Enter dataset name: ")
    df = load_dataset(dataset_name)
    print(df.iloc[:7,:])

if __name__ == "__main__":
    main()