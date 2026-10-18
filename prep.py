
# ============================================================
# PROJECT: VISIT WITH US - MLOPS
# MODULE: DATA PREPROCESSING
#
# Objective:
# 1. Clean customer data
# 2. Select pre-contact prediction features
# 3. Split data into training and testing sets
# 4. Define reusable numerical and categorical transformers
#
# Important:
# The preprocessor is fitted during model training, NOT here.
# This prevents information leakage from the test set.
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

DATA_PATH = Path("tourism.csv")
OUTPUT_DIR = Path("artifacts/processed")

TARGET = "ProdTaken"

# These features are either identifiers or not available
# when the marketing team makes a pre-contact decision.
EXCLUDED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "PitchSatisfactionScore",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "TypeofContact"
]

# Explicit feature lists ensure that a column's meaning
# does not change merely because its storage type changes.
NUMERICAL_FEATURES = [
    "Age",
    "MonthlyIncome",
    "NumberOfTrips",
    "NumberOfPersonVisiting",
    "NumberOfChildrenVisiting"
]

CATEGORICAL_FEATURES = [
    "CityTier",
    "Occupation",
    "Gender",
    "PreferredPropertyStar",
    "MaritalStatus",
    "Passport",
    "OwnCar",
    "Designation"
]

MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ------------------------------------------------------------
# 2. Data Cleaning
# ------------------------------------------------------------

def clean_data(df):
    """
    Apply deterministic data-cleaning rules.

    These rules do not calculate statistics from the data.
    They can therefore be applied consistently to both
    historical training data and new prediction inputs.
    """

    df = df.copy()

    # Remove leading and trailing spaces from text fields.
    text_columns = df.select_dtypes(include=["object"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Standardize known inconsistent categories.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({
            "Fe Male": "Female"
        })

    if "Occupation" in df.columns:
        df["Occupation"] = df["Occupation"].replace({
            "Free Lancer": "Freelancer"
        })

    # We deliberately retain Single and Unmarried
    # until their business meanings are confirmed.

    return df


# ------------------------------------------------------------
# 3. Create the Preprocessing Transformer
# ------------------------------------------------------------

def build_preprocessor():
    """
    Define transformations without fitting them.

    Numerical data:
      - Fill missing values using training median
      - Standardize numerical values

    Categorical data:
      - Fill missing values using most frequent category
      - Convert categories into one-hot encoded indicators
    """

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


# ------------------------------------------------------------
# 4. Prepare Train and Test Data
# ------------------------------------------------------------

def prepare_data():

    # Load the registered source dataset.
    df = pd.read_csv(DATA_PATH)

    # Apply deterministic data cleaning.
    df = clean_data(df)

    # Check that all required predictive fields exist.
    required_columns = MODEL_FEATURES + [TARGET]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Extract the fixed pre-contact feature set.
    X = df[MODEL_FEATURES].copy()

    # Extract the actual purchase outcomes.
    y = df[TARGET].copy()

    # Split data before fitting any learned transformations.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save untransformed train/test sets.
    # Imputation, encoding, and scaling are fitted later
    # inside the model-training pipeline.
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False,
        header=True
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False,
        header=True
    )

    print("Data preprocessing completed.")
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("Numerical features:", NUMERICAL_FEATURES)
    print("Categorical features:", CATEGORICAL_FEATURES)

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# 5. Script Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    prepare_data()
