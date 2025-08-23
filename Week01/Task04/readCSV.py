import pandas as pd

def displaySummary(df):
   print(f"CSV Summary:\n{df.info()}\n\n{df.describe()}")

def main():
   data = pd.read_csv('store.csv')
   df = pd.DataFrame(data)
   displaySummary(df)
   
if __name__ == "__main__":
   main()