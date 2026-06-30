import mlflow
import pandas as pd

mlflow.set_experiment("YouTube Tutorial")

with mlflow.start_run(run_name="Logging Demo", run_id="2fc5532bb489489296de916200e7ec27"):
  # parameters
  # key value
  # dictionary
  
  ####################
  # Logging Parameters
  ####################
  mlflow.log_param("learning_rate", 0.001)
  mlflow.log_param("epochs", 100)
  
  param_dict = {
    "learning_rate1": 0.001,
    "epoch1": 100,
  }
  
  mlflow.log_params(param_dict)
  
  ####################
  # Logging Metrics
  ####################
  mlflow.log_metric("accuracy", 0.9)
  
  metrics_dict = {
    "accuracy1": 0.9,
    "precision": 0.8,
    "recall": 0.7,
    "f1_score": 0.75,
  }
  
  mlflow.log_metrics(metrics_dict)
  
  ####################
  # Logging Artifacts
  ####################
  artifact_path = "loss_curve.png"
  mlflow.log_artifact(artifact_path)
  
  # mlflow.log_image()
  
  demo_df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
  })
  
  titanic_df = pd.read_csv("titanic.csv")
  
  mlflow.log_table(titanic_df, "titanic.json")
  # mlflow.log_assessment(titanic_df, "titanic_assessment.json")
  # mlflow.log_figure(demo_df.plot(kind="bar"), "demo_plot.png")
  

  
  