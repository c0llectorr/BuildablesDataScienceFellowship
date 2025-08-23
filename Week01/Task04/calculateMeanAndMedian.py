import pandas as pd

def calculate_mean_and_median(data):

    series = pd.Series(data)
    mean = series.mean()
    median = series.median()
    return mean, median

def main():
    data = [1, 2, 3, 4, 5]
    mean, median = calculate_mean_and_median(data)
    print(f"Mean: {mean}, Median: {median}")
    
if __name__ == "__main__":
    main()