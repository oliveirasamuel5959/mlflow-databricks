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

mlflow.set_experiment("sklearn model logging")

# with mlflow.start_run(run_name="sklearn model logging"):
#     # Log the hyperparameters
#     mlflow.log_params(params)

#     # Train the model
#     model = LogisticRegression(**params)
#     model.fit(X_train, y_train)
    
#     mlflow.sklearn.log_model(sk_model=model, name="simple_model")
    
mlflow.sklearn.autolog()    
with mlflow.start_run(run_name="sklearn model auto"):
    # Train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
