from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

def drawHistogram(dataset_name):
    available_datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(available_datasets.keys())}.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    df.hist(figsize=(10, 8))
    plt.suptitle(f'Histograms for {dataset_name} dataset', fontsize=16)
    plt.tight_layout()
    plt.show()
    
def main():
    dataset_name = input("Enter dataset name: ")
    drawHistogram(dataset_name)

if __name__ == "__main__":
    main()