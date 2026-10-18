
# ============================================================
# PROJECT: VISIT WITH US - MLOPS
# MODULE: MODEL EVALUATION
#
# Objective:
# Evaluate the selected Random Forest model on unseen
# customer records and measure its marketing usefulness.
#
# Input:
#   X_test.csv
#   y_test.csv
#   best_tourism_model.joblib
#
# Output:
#   evaluation_metrics.json
#   confusion_matrix.png
#   precision_recall_curve.png
#   marketing_lift.csv
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

# ------------------------------------------------------------
# 1. Define Paths
# ------------------------------------------------------------

DATA_DIR = Path("artifacts/processed")
MODEL_DIR = Path("artifacts/models")
REPORT_DIR = Path("artifacts/evaluation")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load Holdout Test Dataset
# ------------------------------------------------------------

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")["ProdTaken"]

# Load the fitted preprocessing + classifier pipeline.
model = joblib.load(
    MODEL_DIR / "best_tourism_model.joblib"
)

print("Test records:", len(X_test))


# ------------------------------------------------------------
# 3. Generate Predictions
# ------------------------------------------------------------

# Predict purchase probabilities.
# [:, 1] selects probability of ProdTaken = 1.
y_proba = model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions at 0.5.
y_pred = (y_proba >= 0.5).astype(int)


# ------------------------------------------------------------
# 4. Calculate Evaluation Metrics
# ------------------------------------------------------------

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "average_precision": average_precision_score(
        y_test, y_proba
    )
}

print("\nHoldout Test Results:")
print(json.dumps(metrics, indent=4))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Save evaluation results.
with open(REPORT_DIR / "evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# ------------------------------------------------------------
# 5. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Purchased", "Purchased"]
).plot(cmap="Blues", values_format="d")

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png")
plt.show()


# ------------------------------------------------------------
# 6. Precision-Recall Curve
# ------------------------------------------------------------

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_proba
)

# Baseline reflects the share of actual buyers.
plt.axhline(
    y_test.mean(),
    linestyle="--",
    label="Baseline purchase rate"
)

plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "precision_recall_curve.png")
plt.show()


# ------------------------------------------------------------
# 7. Marketing Evaluation: Lift at Top K
# ------------------------------------------------------------

# Create a table of actual outcomes and model scores.
ranking = pd.DataFrame({
    "ActualPurchase": y_test.to_numpy(),
    "PredictedProbability": y_proba
})

# Highest purchase probabilities appear first.
ranking = ranking.sort_values(
    "PredictedProbability",
    ascending=False
).reset_index(drop=True)

baseline_rate = ranking["ActualPurchase"].mean()

lift_results = []

# Evaluate several marketing contact budgets.
for percentage in [10, 20, 30, 40, 50]:

    # Calculate the number of customers marketing contacts.
    k = max(1, int(len(ranking) * percentage / 100))

    # Select the highest-ranked K customers.
    top_customers = ranking.head(k)

    # Calculate actual purchase rate among top customers.
    precision_at_k = top_customers["ActualPurchase"].mean()

    # Lift compares selected customers against baseline.
    lift_at_k = precision_at_k / baseline_rate

    # Share of all actual buyers captured in top K.
    recall_at_k = (
        top_customers["ActualPurchase"].sum()
        / ranking["ActualPurchase"].sum()
    )

    lift_results.append({
        "Top_Percent": percentage,
        "Customers_Contacted": k,
        "Precision_At_K": precision_at_k,
        "Recall_At_K": recall_at_k,
        "Lift_At_K": lift_at_k
    })

lift_df = pd.DataFrame(lift_results)

print("\nMarketing Lift Analysis:")
print(lift_df.round(3).to_string(index=False))

lift_df.to_csv(
    REPORT_DIR / "marketing_lift.csv",
    index=False
)

print("\nModel evaluation completed successfully!")
