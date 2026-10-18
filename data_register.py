
# ============================================================
# PROJECT: VISIT WITH US - MLOPS
# MODULE: DATA REGISTRATION
#
# Objective:
# Validate the raw tourism dataset and generate metadata
# to support data traceability and reproducibility.
#
# This module does NOT modify the original dataset.
# ============================================================

import pandas as pd
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

# Define the input dataset and metadata output locations.
DATA_PATH = Path("tourism.csv")
OUTPUT_PATH = Path("artifacts/dataset_metadata.json")

# Define essential dataset columns.
REQUIRED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "Occupation",
    "Gender",
    "MonthlyIncome",
    "NumberOfTrips",
    "Passport",
    "Designation",
    "PitchSatisfactionScore",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "TypeofContact"
]


def register_dataset():

    # 1. Verify that the raw dataset exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # 2. Load raw customer records without modifying them.
    df = pd.read_csv(DATA_PATH)

    # 3. Verify essential columns are present.
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # 4. Validate purchase outcomes.
    # The target must be either 0 or 1.
    if df["ProdTaken"].isna().any() or not (
        df["ProdTaken"].isin([0, 1]).all()
    ):
        raise ValueError("Invalid purchase target values.")

    # 5. Verify customer identifier completeness and uniqueness.
    if df["CustomerID"].isna().any():
        raise ValueError("Missing Customer IDs detected.")

    if df["CustomerID"].duplicated().any():
        raise ValueError("Duplicate Customer IDs detected.")

    # 6. Generate a SHA-256 fingerprint of the raw file.
    # Changes in the dataset will produce a different hash.
    dataset_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    # 7. Prepare registration metadata.
    metadata = {
        "dataset_name": "tourism",
        "registered_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_customers": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "purchase_rate": float(df["ProdTaken"].mean()),
        "sha256": dataset_hash
    }

    # 8. Create the output directory.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 9. Store registration metadata as JSON.
    with open(OUTPUT_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print("Data registration completed successfully!")
    print(json.dumps(metadata, indent=4))


# Execute registration when this file is run directly.
if __name__ == "__main__":
    register_dataset()
