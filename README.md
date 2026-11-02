# IPL Match Winner Classification

This project compares five machine-learning classifiers for predicting whether the team batting first wins an Indian Premier League (IPL) match. It includes reproducible preprocessing, saved model pipelines, a labeled test-data CSV, and a Streamlit evaluation app.

## Problem statement

The goal is to classify a match outcome using information available before play begins. The target variable is `team1_won`:

- `1`: the team batting first won the match.
- `0`: the team batting first did not win the match.

All classifiers are trained on the same training split and evaluated on the same held-out test split. This makes the comparison fair.

## Dataset

Source: [IPL Dataset (2008–2025) on Kaggle](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025)

The original source is ball-by-ball IPL data. To keep the project compact and focused, the preprocessing script converts it into one match-level record and retains the ten most recent completed seasons, 2016–2025.

| Dataset split             | Matches |
| ------------------------- | ------: |
| Clean match-level dataset |     652 |
| Training data             |     521 |
| Test data                 |     131 |

The raw `IPL.csv` file is kept locally and excluded from Git because it is large. The repository includes the reproducible preprocessing script and the labeled `data/test_data.csv` file required for evaluation.

### Features

The models use only pre-match information:

- Batting team and bowling team
- Toss winner and toss decision
- Venue and city
- Match stage
- Year, month, day, and day of week

Post-match information such as scores, wickets, player of the match, and winning margin is excluded to prevent target leakage.

## Models

The following classifiers are trained and saved in `models/`:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine

## Model comparison

All values below were calculated using the same 131-match test set.

| Model                  | Accuracy |    AUC | Precision | Recall | F1 Score |     MCC |
| ---------------------- | -------: | -----: | --------: | -----: | -------: | ------: |
| Logistic Regression    |   0.5191 | 0.5083 |    0.4545 | 0.4310 |   0.4425 |  0.0202 |
| Decision Tree          |   0.5344 | 0.5202 |    0.4694 | 0.3966 |   0.4299 |  0.0415 |
| K-Nearest Neighbors    |   0.4427 | 0.4049 |    0.3333 | 0.2586 |   0.2913 | -0.1593 |
| Gaussian Naive Bayes   |   0.4656 | 0.5030 |    0.4444 | 0.8276 |   0.5783 |  0.0074 |
| Support Vector Machine |   0.5573 | 0.5475 |    0.0000 | 0.0000 |   0.0000 |  0.0000 |

## Observations

- **Logistic Regression:** a simple baseline with performance close to random guessing. It identifies both classes, but its overall signal is weak.
- **Decision Tree:** achieved the highest MCC and is the most balanced model in this comparison because it makes useful predictions for both target classes.
- **K-Nearest Neighbors:** produced the lowest accuracy and a negative MCC, indicating weaker performance than the other models.
- **Gaussian Naive Bayes:** achieved the highest recall and F1 score. However, its lower precision and near-zero MCC indicate that it predicts many false wins.
- **Support Vector Machine:** achieved the highest accuracy and AUC, but predicted only the majority class on this test set. It therefore has zero precision, recall, F1 score, and MCC for the positive class.

**Overall conclusion:** Decision Tree is selected as the most balanced model because it has the highest MCC and makes non-trivial predictions for both classes. The results also show that basic pre-match match metadata is not enough for strong prediction; historical team-form features would be a valuable next improvement.

## Project structure

```text
.
├── app.py                         # Streamlit evaluation application
├── data/
│   ├── processed/                  # Clean match-level dataset
│   ├── train_data.csv              # Training split
│   └── test_data.csv               # Labeled test split for evaluation
├── models/                         # Saved .joblib classifier pipelines
├── reports/                        # Metric CSV files for all five models
├── src/
│   ├── prepare_match_data.py
│   └── *_training.py
└── requirements.txt
```

## Installation

```bash
git clone git@github.com:sandipnirmal/match_winner_prediction_ml_assignment.git
cd match_winner_prediction_ml_assignment
python3 -m pip install -r requirements.txt
```

## Reproduce the data split

Place the original Kaggle `IPL.csv` file in `data/`, then run:

```bash
python3 src/prepare_match_data.py
```

## Train the models

```bash
python3 src/logistic_regression_training.py
python3 src/decision_tree_classifier_training.py
python3 src/k_nearest_neighbour_classifier_training.py
python3 src/naive_bayes_classifier_training.py
python3 src/support_vector_machine_training.py
```

Each script saves its trained pipeline in `models/` and its required metrics in `reports/`.

## Run the Streamlit application

```bash
streamlit run app.py
```

In the app, choose a saved classifier and upload `data/test_data.csv`. The app displays the six evaluation metrics, a confusion matrix, and a classification report without retraining the model.

## Streamlit application

- Streamlit Community Cloud hosted at -> https://2026ak05319-sandip-ml.streamlit.app/
