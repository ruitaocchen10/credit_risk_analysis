import pandas as pd

# initial data profiling
def profile_dataset(df, dataset_name):
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ASSESSMENT: {dataset_name}")
    print(f"{'='*60}\n")

    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns") #shape
    print(f"Memory Usage: {df.memory_usage(deep=True).sum()} B") #calculate memory usage
    print("\n")

    print(df.dtypes.value_counts()) #getting value counts of the different data types
    print("\n")

    print(f"Duplicate Rows: {df.duplicated().sum()}") #duplicate check
    print("\n")

    print(df.isnull().sum()) #null check
    print("\n")

df = pd.read_csv("/Users/ruitao/credit_risk_analysis/data/raw/application_train.csv")
profile_dataset(df, "Application Train")