
# ============================================================
# PROJECT: VISIT WITH US - MLOPS
# MODULE: MODEL TRAINING
#
# Business Objective:
# Predict the probability of a customer purchasing
# a travel package before the marketing team contacts them.
#
# Technical Objective:
# 1. Load the preprocessed training dataset.
# 2. Train three classification algorithms.
# 3. Evaluate models using stratified cross-validation.
# 4. Select the best model using Average Precision.
# 5. Save the complete fitted model pipeline.
#
# Important:
# The holdout test dataset is NOT used in this stage.
# ============================================================

from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Import the SAME preprocessing function from prep.py.
from prep import build_preprocessor


# ------------------------------------------------------------
# 1. Configure Input and Output Paths
# ------------------------------------------------------------

DATA_DIR = Path("artifacts/processed")
MODEL_DIR = Path("artifacts/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ------------------------------------------------------------
# 2. Load Training Data
# ------------------------------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")

# Convert target column into a one-dimensional Series.
y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
)["ProdTaken"]

print("Training dataset:", X_train.shape)
print("Target distribution:")
print(y_train.value_counts(normalize=True))


# ------------------------------------------------------------
# 3. Define Machine-Learning Algorithms
# ------------------------------------------------------------

models = {

    # Baseline classification algorithm.
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    # Ensemble of decision trees.
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    # Gradient-boosted decision-tree algorithm.
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ------------------------------------------------------------
# 4. Configure Cross-Validation
# ------------------------------------------------------------

# Create five stratified folds from the training dataset.
# Each fold contains approximately the same purchase ratio.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Define evaluation metrics.
scoring = {
    "PR_AUC": "average_precision",
    "ROC_AUC": "roc_auc",
    "F1": "f1",
    "Precision": "precision",
    "Recall": "recall"
}


# ------------------------------------------------------------
# 5. Train and Compare Models
# ------------------------------------------------------------

results = []

for model_name, classifier in models.items():

    print(f"\nTraining: {model_name}")

    # Bundle preprocessing and classification.
    # Each cross-validation fold fits preprocessing
    # using only that fold's training observations.
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier)
    ])

    # Evaluate on five training-data folds.
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    # Store mean validation metrics across the five folds.
    result = {
        "Model": model_name,
        "PR_AUC": scores["test_PR_AUC"].mean(),
        "ROC_AUC": scores["test_ROC_AUC"].mean(),
        "Precision": scores["test_Precision"].mean(),
        "Recall": scores["test_Recall"].mean(),
        "F1": scores["test_F1"].mean()
    }

    results.append(result)

    print("PR-AUC:", round(result["PR_AUC"], 4))
    print("ROC-AUC:", round(result["ROC_AUC"], 4))


# ------------------------------------------------------------
# 6. Select the Best Model
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

# Select model with the highest average precision.
results_df = results_df.sort_values(
    by="PR_AUC",
    ascending=False
).reset_index(drop=True)

print("\nModel comparison:")
print(results_df.round(4).to_string(index=False))

best_model_name = results_df.loc[0, "Model"]

print("\nSelected model:", best_model_name)


# ------------------------------------------------------------
# 7. Fit the Selected Model on All Training Data
# ------------------------------------------------------------

# Build a fresh pipeline for the winning algorithm.
best_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("classifier", models[best_model_name])
])

# Fit all preprocessing steps and the classifier.
best_pipeline.fit(X_train, y_train)


# ------------------------------------------------------------
# 8. Save Trained Pipeline and Results
# ------------------------------------------------------------

# Save the entire pipeline, including the fitted
# imputer, scaler, encoder and classification model.
joblib.dump(
    best_pipeline,
    MODEL_DIR / "best_tourism_model.joblib"
)

# Save the cross-validation comparison.
results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)

print("\nTraining completed successfully!")
print("Model saved to:", MODEL_DIR / "best_tourism_model.joblib")
