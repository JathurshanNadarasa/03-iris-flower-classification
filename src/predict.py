import joblib
import pandas as pd


# ==========================================
# 1. Load Saved Model
# ==========================================

model = joblib.load(
    "./models/iris_model.pkl"
)


# ==========================================
# 2. Get Flower Measurements
# ==========================================

sepal_length = float(
    input("Enter sepal length (cm): ")
)

sepal_width = float(
    input("Enter sepal width (cm): ")
)

petal_length = float(
    input("Enter petal length (cm): ")
)

petal_width = float(
    input("Enter petal width (cm): ")
)


# ==========================================
# 3. Create Input DataFrame
# ==========================================

flower = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)


# ==========================================
# 4. Make Prediction
# ==========================================

prediction = model.predict(
    flower
)


# ==========================================
# 5. Get Prediction Probabilities
# ==========================================

probabilities = model.predict_proba(
    flower
)


# ==========================================
# 6. Get Class Names
# ==========================================

classes = model.classes_


# ==========================================
# 7. Display Result
# ==========================================

print("\n==============================")
print("Iris Flower Prediction")
print("==============================")

print(
    "Predicted species:",
    prediction[0]
)

print("\nPrediction Probabilities")
print("========================")

for class_name, probability in zip(
    classes,
    probabilities[0]
):
    print(
        f"{class_name}: {probability * 100:.2f}%"
    )