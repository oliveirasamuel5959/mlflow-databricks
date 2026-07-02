import mlflow

mlflow.set_experiment("demo experiment")

with mlflow.start_run():
  mlflow.log_params({"param1": 5, "param2": "value2"})