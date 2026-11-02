"""Streamlit app for evaluating saved IPL match-winner classifiers."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "team1_won"

CATEGORICAL_FEATURES = [
    "batting_team", "bowling_team", "toss_winner", "toss_decision", "venue", "city", "stage"
]
NUMERIC_FEATURES = ["year", "month", "day", "day_of_week"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbour.joblib",
    "Gaussian Naive Bayes": "naive_bayes_classifier.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(model_filename: str):
    """Load a saved model once and reuse it while the app is running."""
    return joblib.load(PROJECT_DIR / "models" / model_filename)


def calculate_metrics(actual_values, predictions, probabilities) -> dict:
    """Calculate the six metrics required for the assignment."""
    return {
        "Accuracy": accuracy_score(actual_values, predictions),
        "AUC": roc_auc_score(actual_values, probabilities),
        "Precision": precision_score(actual_values, predictions, zero_division=0),
        "Recall": recall_score(actual_values, predictions, zero_division=0),
        "F1 Score": f1_score(actual_values, predictions, zero_division=0),
        "MCC": matthews_corrcoef(actual_values, predictions),
    }


def main() -> None:
    """Render a model selector and evaluate an uploaded labeled test CSV."""
    st.set_page_config(page_title="IPL Match Winner Classifier", page_icon="🏏")
    st.title("IPL Match Winner Classifier")
    st.caption("Evaluate saved models without retraining them.")

    selected_model = st.selectbox("Choose a classifier", list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload labeled test data", type="csv")

    with st.expander("Required CSV columns"):
        st.code(", ".join(FEATURE_COLUMNS + [TARGET_COLUMN]), language=None)
        st.write("`team1_won` must be 1 when the first-batting team won, otherwise 0.")

    if uploaded_file is None:
        st.info("Upload `data/test_data.csv` to evaluate a selected saved model.")
        return

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"The uploaded file could not be read as CSV: {error}")
        return

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in uploaded_data]
    if missing_columns:
        st.error("The uploaded CSV is missing: " + ", ".join(missing_columns))
        return

    if uploaded_data[TARGET_COLUMN].nunique() < 2:
        st.error("The uploaded CSV needs both target classes (0 and 1) to calculate AUC.")
        return

    try:
        model = load_model(MODEL_FILES[selected_model])
    except FileNotFoundError:
        st.error("Saved model file not found. Run its training script first.")
        return

    features = uploaded_data[FEATURE_COLUMNS]
    actual_values = uploaded_data[TARGET_COLUMN]
    predictions = model.predict(features)
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, positive_class_index]
    metrics = calculate_metrics(actual_values, predictions, probabilities)

    st.subheader(f"Results: {selected_model}")
    metric_columns = st.columns(3)
    for column, (metric_name, metric_value) in zip(metric_columns * 2, metrics.items()):
        column.metric(metric_name, f"{metric_value:.4f}")

    st.subheader("Confusion Matrix")
    figure, axis = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        actual_values, predictions,
        display_labels=["Team 1 did not win", "Team 1 won"],
        cmap="Blues", ax=axis,
    )
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Classification Report")
    report = classification_report(
        actual_values, predictions,
        target_names=["Team 1 did not win", "Team 1 won"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("Uploaded Test Data")
    st.dataframe(uploaded_data, use_container_width=True)


if __name__ == "__main__":
    main()
