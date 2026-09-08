from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df  = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

df['species'] = df["species"].map({
    0:'setosa',
    1:'versicolor',
    2:'virginica'
})

df.to_csv(
    './data/iris.csv',
    index = False

)

print("Iris dataset created successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nSpecies:")
print(df["species"].value_counts())