import mlflow

mlflow.set_experiment("nested run demo")

with mlflow.start_run(run_name="Parent Run") as parent_run:
  print(f"Parent run: {parent_run.info.run_id}")
  mlflow.log_param("theta", 100)
  
  with mlflow.start_run(run_name="Child Run 1", nested=True) as child_run1:
    print(f"Child run: {child_run1.info.run_id}")
  
  with mlflow.start_run(run_name="Child Run 2", nested=True) as child_run2:
    print(f"Child run: {child_run2.info.run_id}")
  
  with mlflow.start_run(run_name="Child Run 3", nested=True) as child_run3:
    print(f"Child run: {child_run3.info.run_id}")