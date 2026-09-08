import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

df = pd.read_csv("./data/iris.csv")


X = df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
]


y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)

print("Random Forest Iris Classification")
print("=================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAccuracy:", accuracy)

print("\nActual vs Predicted")
print("===================")

for actual, predicted in zip(y_test, predictions):
    print(
        f"Actual: {actual:12} | Predicted: {predicted}"
    )


cm = confusion_matrix(
    y_test,
    predictions,
    labels=[
        "setosa",
        "versicolor",
        "virginica"
    ]
)

print("\nConfusion Matrix")
print("================")

print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "setosa",
        "versicolor",
        "virginica"
    ]
)

print("\nClassification Report")
print("=====================")

print(
    classification_report(
        y_test,
        predictions
    )
)

display.plot()

plt.title("Random Forest Confusion Matrix")

plt.show()