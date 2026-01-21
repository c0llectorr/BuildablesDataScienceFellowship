import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import pandas as pd

def splitDataset(dataset_name, test_size, random_state):
    available_datasets = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(available_datasets.keys())}.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_set, test_set

def main():
    dataset_name = input("Enter dataset name: ")
    test_size = float(input("Enter test size (0.1 - 0.9): "))
    random_state = int(input("Enter random state (integer): "))
    
    train_set, test_set = splitDataset(dataset_name, test_size, random_state)
    
    print(f"Training set shape: {train_set.shape}")
    print(f"Testing set shape: {test_set.shape}")
    
if __name__ == "__main__":
    main()