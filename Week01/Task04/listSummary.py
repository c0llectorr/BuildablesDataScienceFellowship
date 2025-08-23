import pandas as pd
import numpy as np
    
def main():
    data = [input("Enter numbers separated by space: ").split(" ")]
    data = np.array(data, dtype=int).flatten()
    sf = pd.Series(data)
    print(sf.describe())
    
    
if __name__ == "__main__":
    main()