"""This is Decision Tree Classifier on IPL data"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = PROJECT_DIR / "data" / "test_data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "decision_tree.joblib"
METRICS_PATH = PROJECT_DIR / "reports" / "decision_tree_metrics.csv"

TARGET_COLUMN = "team1_won"

CATEGORICAL_FEATURES = [
    "batting_team",
    "bowling_team",
    "toss_winner",
    "toss_decision",
    "venue",
    "city",
    "stage",
]
NUMERIC_FEATURES = ["year", "month", "day", "day_of_week"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def create_model_pipeline() -> Pipeline:
    """Create reusable pipeline for Decision Tree Classifier"""
    preprocess_features = ColumnTransformer(
        transformers=[
            (
                "categories",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numbers",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )

    classifier = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess_features),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    actual_values: pd.Series, predictions: pd.Series, probabilities: pd.Series
) -> dict:
    """Calculate all required metrics"""
    return {
        "accuracy": accuracy_score(actual_values, predictions),
        "auc": roc_auc_score(actual_values, probabilities),
        "precision": precision_score(actual_values, predictions, zero_division=0),
        "recall": recall_score(actual_values, predictions, zero_division=0),
        "f1_score": f1_score(actual_values, predictions, zero_division=0),
        "mcc": matthews_corrcoef(actual_values, predictions),
    }


def main() -> None:
    """Fit the model, print metrics and save for Streamlit app"""
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    model = create_model_pipeline()
    model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])

    predictions = model.predict(test_data[FEATURE_COLUMNS])
    win_probabilities = model.predict_proba(test_data[FEATURE_COLUMNS])[:, 1]
    metrics = calculate_metrics(
        test_data[TARGET_COLUMN], predictions, win_probabilities
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    print("Decision Tree Classifier metrics")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value: .4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
