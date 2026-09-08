import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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
# 3. Create Models
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=200
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 4. Create Cross-Validation
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 5. Evaluate Models
# ==========================================

print("Scaled Cross-Validation Results")
print("================================")


for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"\n{name}")

    print("Fold accuracies:")

    for i, score in enumerate(scores, start=1):

        print(
            f"Fold {i}: {score:.4f}"
        )

    print(
        "Average accuracy:",
        f"{scores.mean():.4f}"
    )

    print(
        "Standard deviation:",
        f"{scores.std():.4f}"
    )