# Iris Flower Classification

A machine learning classification project that predicts the species of an Iris flower based on its sepal and petal measurements.

This project explores multiple classification algorithms, compares their performance, applies feature scaling and a machine learning pipeline, and exposes the final model through a FastAPI REST API.

## Project Overview

The model predicts one of three Iris flower species:

* Setosa
* Versicolor
* Virginica

The prediction is based on four measurements:

* Sepal length
* Sepal width
* Petal length
* Petal width

Several classification algorithms are evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

After comparing the models and performing cross-validation, the selected model is **K-Nearest Neighbors (KNN) with StandardScaler**, using `K=5`.

> **Note:** This project uses the well-known Iris dataset for machine learning practice and education.

## Features

* Generate the Iris dataset
* Explore the dataset using Pandas
* Analyze dataset shape, columns, data types, and statistics
* Check for missing values
* Visualize the dataset
* Train multiple classification models
* Compare model performance
* Generate confusion matrices
* Generate classification reports
* Perform cross-validation
* Apply feature scaling using StandardScaler
* Build a Scikit-learn Pipeline
* Save and load the trained model using Joblib
* Generate prediction probabilities
* Make individual flower predictions
* Provide a FastAPI REST API
* Validate API input using Pydantic
* Provide API health checking
* Test predictions using Swagger UI

## Dataset

The dataset contains Iris flower measurements and their corresponding species.

### Input Features

| Feature        | Description         |
| -------------- | ------------------- |
| `sepal_length` | Length of the sepal |
| `sepal_width`  | Width of the sepal  |
| `petal_length` | Length of the petal |
| `petal_width`  | Width of the petal  |

### Target

```text
species
```

The target contains three classes:

```text
setosa
versicolor
virginica
```

## Machine Learning Workflow

```text
Iris Dataset
     ↓
Data Exploration
     ↓
Data Visualization
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Multiple Classification Models
     ↓
Model Comparison
     ↓
Confusion Matrix
     ↓
Classification Report
     ↓
Cross Validation
     ↓
Feature Scaling
     ↓
Pipeline
     ↓
Select KNN Model
     ↓
Save Model
     ↓
Prediction
     ↓
FastAPI REST API
```

## Models Used

### 1. Logistic Regression

A linear classification algorithm used as one of the baseline models.

### 2. K-Nearest Neighbors

KNN classifies a flower based on the classes of nearby observations.

The final model uses:

```text
StandardScaler + KNN
K = 5
```

### 3. Decision Tree

A tree-based classification algorithm that makes decisions using feature-based conditions.

### 4. Random Forest

An ensemble model consisting of multiple decision trees.

## Model Selection

The models were compared using classification performance and cross-validation.

The final selected model was:

```text
K-Nearest Neighbors
+
StandardScaler
+
K = 5
```

The scaled KNN model achieved approximately **97.33% mean accuracy in 5-fold cross-validation** during the project evaluation.

## Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Pandas       | Data manipulation and analysis |
| NumPy        | Numerical operations           |
| Matplotlib   | Data visualization             |
| Seaborn      | Statistical visualization      |
| Scikit-learn | Machine learning               |
| Joblib       | Model saving and loading       |
| FastAPI      | REST API                       |
| Pydantic     | API input validation           |
| Uvicorn      | ASGI server                    |

## Project Structure

```text
03-iris-flower-classification/
│
├── data/
│   └── iris.csv
│
├── models/
│   └── .gitkeep
│
├── src/
│   ├── create_dataset.py
│   ├── explore.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/JathurshanNadarasa/03-iris-flower-classification.git
```

Navigate into the project:

```bash
cd 03-iris-flower-classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Generate the Dataset

Run:

```bash
python src/create_dataset.py
```

This creates:

```text
data/iris.csv
```

## Explore the Dataset

Run:

```bash
python src/explore.py
```

The exploration script provides information about:

* Dataset records
* Dataset shape
* Feature names
* Data types
* Missing values
* Statistical summary
* Feature relationships
* Visualizations

## Train the Model

Run:

```bash
python src/train.py
```

The training process includes the classification models used in the project and saves the trained model for later prediction.

The saved model is stored inside:

```text
models/
```

## Evaluate the Models

Run:

```bash
python src/evaluate.py
```

The evaluation process includes:

* Model accuracy
* Confusion matrix
* Classification report
* Model comparison

## Cross Validation

The project uses cross-validation to provide a more reliable estimate of model performance.

The final scaled KNN model achieved approximately:

```text
97.33% mean accuracy
```

using 5-fold cross-validation.

## Make a Prediction

Run:

```bash
python src/predict.py
```

The prediction process accepts Iris flower measurements and predicts the species.

The model can also provide prediction probabilities for the possible classes.

## FastAPI

The project includes a REST API for making Iris flower predictions.

Start the API:

```bash
uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## API Endpoints

### GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "Iris Flower Classification API is running"
}
```

### GET `/health`

Checks the health of the API and model.

Example response:

```json
{
  "status": "healthy",
  "model": "iris_classification_model"
}
```

### POST `/predict`

Predicts the Iris flower species.

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "predicted_species": "setosa"
}
```

The API uses Pydantic validation to prevent invalid measurement values from being submitted.

## Swagger API Documentation

FastAPI automatically provides interactive API documentation.

After starting the API, open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI allows you to test the `/predict` endpoint directly from the browser.

## Feature Scaling

The project uses `StandardScaler` before KNN classification.

The purpose of scaling is to put the numerical features on a comparable scale so that features with larger numerical ranges do not disproportionately influence distance-based algorithms such as KNN.

The final workflow uses:

```text
StandardScaler
      ↓
KNeighborsClassifier
```

This is implemented using a Scikit-learn Pipeline.

## Model Persistence

The trained model is saved using Joblib so it can be reused without retraining every time.

Example:

```python
joblib.dump(model, "models/iris_model.pkl")
```

The saved model is excluded from Git using `.gitignore`.

## Learning Outcomes

This project provided practical experience with:

* Classification
* Pandas
* NumPy
* Exploratory Data Analysis
* Data visualization
* Feature selection
* Logistic Regression
* KNN
* Decision Trees
* Random Forest
* Model comparison
* Confusion matrices
* Classification reports
* Cross-validation
* Feature scaling
* StandardScaler
* Scikit-learn Pipeline
* Model persistence
* Joblib
* Prediction probabilities
* FastAPI
* Pydantic validation
* REST APIs
* Swagger documentation

## Future Improvements

Possible improvements include:

* Add a web-based frontend
* Deploy the API
* Add automated unit tests
* Add Docker support
* Add model monitoring
* Experiment with additional classification algorithms
* Add more detailed visualizations
* Create a user-friendly flower prediction interface

## Author

**Jathurshan Nadarasa**

Bachelor of Information Technology

This project was developed as part of a practical machine learning portfolio.
