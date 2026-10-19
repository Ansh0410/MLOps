
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

MODEL_FILE = Path(
    "artifacts/models/best_tourism_model.joblib"
)

DATASET_METADATA = Path(
    "artifacts/dataset_metadata.json"
)

COMPARISON_FILE = Path(
    "artifacts/models/model_comparison.csv"
)

OUTPUT_FILE = Path(
    "reports/model_manifest.json"
)


def calculate_sha256(file_path):
    """Generate a fingerprint of a file's exact contents."""
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# Confirm that our training stage produced a model.
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Trained model missing: {MODEL_FILE}"
    )

if not DATASET_METADATA.exists():
    raise FileNotFoundError(
        "Dataset registration metadata missing."
    )

# Read previously generated metadata.
with open(DATASET_METADATA, encoding="utf-8") as file:
    dataset_metadata = json.load(file)

manifest = {
    "project": "Visit with Us",
    "model_filename": MODEL_FILE.name,
    "model_sha256": calculate_sha256(MODEL_FILE),
    "dataset_metadata_sha256": calculate_sha256(
        DATASET_METADATA
    ),
    "source_git_commit": os.getenv(
        "GITHUB_SHA", "local_colab_run"
    ),
    "github_run_id": os.getenv(
        "GITHUB_RUN_ID", "local"
    ),
    "github_run_number": os.getenv(
        "GITHUB_RUN_NUMBER", "local"
    ),
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "model_selection_metric": "CV average precision (PR-AUC)"
}

# Add the model comparison file fingerprint.
if COMPARISON_FILE.exists():
    manifest["model_comparison_sha256"] = (
        calculate_sha256(COMPARISON_FILE)
    )

# Record a reference to source dataset registration.
# Avoid copying customer-level information.
manifest["dataset_registration"] = dataset_metadata

# Create a version identifier tied to a specific workflow run.
manifest["version_id"] = (
    f"run-{manifest['github_run_id']}-"
    f"{manifest['model_sha256'][:12]}"
)

OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)

OUTPUT_FILE.write_text(
    json.dumps(manifest, indent=2, default=str),
    encoding="utf-8"
)

print("Model manifest created:", OUTPUT_FILE)
print("Model version:", manifest["version_id"])
print("Git commit:", manifest["source_git_commit"])
