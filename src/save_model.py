import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("./data/iris.csv")


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
]

y = df["species"]


# ==========================================
# 3. Create KNN Pipeline
# ==========================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5
    ))
])


# ==========================================
# 4. Train Model
# ==========================================

model.fit(
    X,
    y
)


# ==========================================
# 5. Save Model
# ==========================================

joblib.dump(
    model,
    "./models/iris_model.pkl"
)


# ==========================================
# 6. Success Message
# ==========================================

print("Iris model trained successfully!")

print("Model: KNN")
print("Scaler: StandardScaler")
print("K value: 5")

print("\nModel saved successfully!")
print("Location: models/iris_model.pkl")