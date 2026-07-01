import mlflow

model_uri = "models:/best-production-model/2"

mlflow_model = mlflow.sklearn.load_model(model_uri=model_uri)

print(mlflow_model)