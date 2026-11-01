"""Create a clean match-level dataset for IPL winner prediction.

The Kaggle source is ball-by-ball data.  This script keeps one record per
match and deliberately uses only information available before a match starts.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "IPL.csv"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "ipl_matches_2016_2025.csv"
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = PROJECT_DIR / "data" / "test_data.csv"

FIRST_SEASON = 2016
LAST_SEASON = 2025
TEST_SIZE = 0.20
RANDOM_STATE = 42


def build_match_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per completed IPL match with pre-match features only."""
    ipl_matches = raw_data.loc[
        (raw_data["event_name"] == "Indian Premier League")
        & raw_data["year"].between(FIRST_SEASON, LAST_SEASON)
        & (raw_data["innings"] == 1)
    ].copy()

    # A match has many deliveries.  Its first delivery carries the match-level
    # details that are known before play starts, so retain one row per match.
    matches = (
        ipl_matches.sort_values(["match_id", "ball_no"])
        .drop_duplicates(subset="match_id", keep="first")
        .copy()
    )

    # Matches with no winner (for example, abandoned games) cannot be used for
    # a binary winner classifier.
    matches = matches[matches["match_won_by"].notna()].copy()
    matches = matches[matches["match_won_by"] != "NA"].copy()

    matches["match_date"] = pd.to_datetime(matches["date"])
    matches["day_of_week"] = matches["match_date"].dt.dayofweek
    matches["team1_won"] = (matches["match_won_by"] == matches["batting_team"]).astype(int)

    selected_columns = [
        "match_id",
        "match_date",
        "batting_team",
        "bowling_team",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",
        "year",
        "month",
        "day",
        "day_of_week",
        "stage",
        "team1_won",
    ]
    return matches[selected_columns].reset_index(drop=True)


def main() -> None:
    """Build and save reproducible training and test CSV files."""
    raw_data = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    matches = build_match_dataset(raw_data)

    train_data, test_data = train_test_split(
        matches,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=matches["team1_won"],
    )

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED_DATA_PATH, index=False)
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    print(f"Created {len(matches)} match records from {FIRST_SEASON} to {LAST_SEASON}.")
    print(f"Training records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")
    print(f"Saved processed data to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
