from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd

def statisticalAnalysis(dataset_name):
    available_datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(available_datasets.keys())}.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    std = df.std()
    mean = df.mean()
    median = df.median()
    variance = df.var()
    
    print(
        f"\nStandard Deviation:\n{std}\n\n"
        f"Mean:\n{mean}\n\n"
        f"Median:\n{median}\n\n"
        f"Variance:\n{variance}"
        )

def main():
    dataset_name = input("Enter dataset name: ")
    print(statisticalAnalysis(dataset_name))
    
if __name__ == "__main__":
    main()