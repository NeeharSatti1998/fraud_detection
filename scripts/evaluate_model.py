import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil
import tempfile

EXPERIMENT_NAME = "Invoice_fraud_detection"
PROMOTION_DIR = "model"
PROMOTION_NAME = "production_model.pkl"
METRIC = "roc_auc"


def promote_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC} DESC"]
    )

    if not runs:
        raise Exception("No completed runs found.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    print(f" Best run: {best_run_id} with {METRIC} = {best_run.data.metrics.get(METRIC)}")


    artifacts = client.list_artifacts(best_run_id)
    model_artifact = next((a for a in artifacts if a.path.endswith(".pkl")), None)

    if not model_artifact:
        raise Exception("No .pkl model artifact found.")
    
    with tempfile.TemporaryDirectory() as tmp:
        local_model_path = client.download_artifacts(best_run_id, model_artifact.path, dst_path=tmp)


        promoted_path = os.path.join(PROMOTION_DIR, PROMOTION_NAME)
        os.makedirs(PROMOTION_DIR, exist_ok=True)
        shutil.copy(local_model_path, promoted_path)
        print(f"Promoted model saved to {promoted_path}")

if __name__ == "__main__":
    promote_best_model()