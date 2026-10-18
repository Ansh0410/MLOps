
# ============================================================
# PROJECT: VISIT WITH US - MLOPS
# MODULE: MLFLOW EXPERIMENT TRACKING
#
# Objective:
# Record the completed model comparison, selected model,
# dataset fingerprint and final evaluation artifacts.
#
# This script logs existing results. It does not retrain
# or recalculate any models.
# ============================================================

import json
from pathlib import Path

import pandas as pd
import mlflow


# ------------------------------------------------------------
# 1. Configure Artifact Paths
# ------------------------------------------------------------

BASE_DIR = Path("artifacts")

COMPARISON_PATH = (
    BASE_DIR / "models" / "model_comparison.csv"
)

MODEL_PATH = (
    BASE_DIR / "models" / "best_tourism_model.joblib"
)

EVALUATION_DIR = BASE_DIR / "evaluation"

METADATA_PATH = BASE_DIR / "dataset_metadata.json"


# ------------------------------------------------------------
# 2. Configure MLflow
# ------------------------------------------------------------

# Store MLflow tracking metadata in a local SQLite database.
# Files and model artifacts remain in local artifact storage.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Group related model runs under one experiment.
mlflow.set_experiment(
    "Visit_With_Us_Purchase_Prediction"
)


# ------------------------------------------------------------
# 3. Load Existing Experiment Results
# ------------------------------------------------------------

comparison = pd.read_csv(COMPARISON_PATH)

with open(METADATA_PATH, "r") as f:
    dataset_metadata = json.load(f)

with open(EVALUATION_DIR / "evaluation_metrics.json", "r") as f:
    evaluation_metrics = json.load(f)

# The winning model was selected using validation PR-AUC.
best_model_name = comparison.sort_values(
    "PR_AUC",
    ascending=False
).iloc[0]["Model"]


# ------------------------------------------------------------
# 4. Log All Three Model Experiments
# ------------------------------------------------------------

for _, result in comparison.iterrows():

    model_name = str(result["Model"])

    # Each model gets a separate MLflow run.
    with mlflow.start_run(run_name=model_name):

        # Record dataset identity.
        mlflow.log_param(
            "dataset_sha256",
            dataset_metadata["sha256"]
        )

        mlflow.log_param(
            "dataset_rows",
            dataset_metadata["total_customers"]
        )

        # Record the algorithm and evaluation method.
        mlflow.log_param("algorithm", model_name)
        mlflow.log_param("cv_folds", 5)

        # Record mean cross-validation metrics.
        # Prefix indicates that these are validation metrics,
        # not final holdout test results.
        for metric in [
            "PR_AUC",
            "ROC_AUC",
            "Precision",
            "Recall",
            "F1"
        ]:
            mlflow.log_metric(
                f"cv_{metric.lower()}",
                float(result[metric])
            )

        # Identify which model was selected.
        is_best = model_name == best_model_name

        mlflow.set_tag(
            "selected_model",
            str(is_best).lower()
        )

        mlflow.set_tag(
            "project",
            "Visit_With_Us"
        )

        mlflow.set_tag(
            "run_type",
            "historical_results_import"
        )

        # Only the winning model was fitted and saved
        # on the full training dataset in our train.py.
        if is_best:

            # Store final holdout metrics.
            for metric, value in evaluation_metrics.items():
                mlflow.log_metric(
                    f"test_{metric}",
                    float(value)
                )

            # Preserve the selected trained pipeline.
            mlflow.log_artifact(
                str(MODEL_PATH),
                artifact_path="model"
            )

            # Preserve data registration metadata.
            mlflow.log_artifact(
                str(METADATA_PATH),
                artifact_path="dataset"
            )

            # Preserve evaluation figures and reports.
            mlflow.log_artifacts(
                str(EVALUATION_DIR),
                artifact_path="evaluation"
            )

        print("Logged:", model_name)

print("\nMLflow experiment tracking completed!")
