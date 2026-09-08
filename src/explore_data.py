import pandas as pd

df = pd.read_csv("./data/iris.csv")

print("First 5 rows")
print("========")
print(df.head())


print("\nDataset Information")
print("========")
print(df.info())

print("\nStatistical Summary")
print("========")
print(df.describe())

print("\nMissing Values")
print("=========")
print(df.isnull().sum())


print("\nSpecies Count")
print("========")
print(df['species'].value_counts())