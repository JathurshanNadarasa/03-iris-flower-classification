import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd


# ==========================================
# 1. Create FastAPI Application
# ==========================================

app = FastAPI(
    title="Iris Flower Classification API",
    description="ML API for predicting Iris flower species",
    version="1.0.0"
)


# ==========================================
# 2. Load Saved ML Model
# ==========================================

model = joblib.load(
    "models/iris_model.pkl"
)


# ==========================================
# 3. Define Input Schema
# ==========================================

class IrisInput(BaseModel):

    sepal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="Sepal length in centimeters"
    )

    sepal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="Sepal width in centimeters"
    )

    petal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="Petal length in centimeters"
    )

    petal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="Petal width in centimeters"
    )


# ==========================================
# 4. Health Check
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Iris Classification API is running"
    }


# ==========================================
# 5. Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict(data: IrisInput):

    flower = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    # Make prediction
    prediction = model.predict(
        flower
    )

    # Get probabilities
    probabilities = model.predict_proba(
        flower
    )

    # Get class names
    classes = model.classes_

    probability_result = {}

    for class_name, probability in zip(
        classes,
        probabilities[0]
    ):

        probability_result[class_name] = round(
            float(probability),
            4
        )

    return {
        "prediction": prediction[0],
        "probabilities": probability_result
    }