import pandas as pd
import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 800,
    "random_state": 8888,
}

mlflow.set_experiment("demo experiment")

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log the evaluation metrics
    mlflow.log_metrics({
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1_score": f1,
    })
    
    mlflow.sklearn.log_model(model, "best-production-model")
    
    mlflow.log_artifact("lect5.py", artifact_path="source_code")