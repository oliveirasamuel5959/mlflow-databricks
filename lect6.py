import mlflow

model_uri = "models:/best-production-model/1"

mlflow_model = mlflow.sklearn.load_model(model_uri)

print("Loaded model from MLflow Model Registry:")
print(mlflow_model)