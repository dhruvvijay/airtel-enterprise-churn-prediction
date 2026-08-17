"""
data_preprocessing.py
----------------------
Data loading, validation, and cleaning for the Airtel churn dataset.
Kept separate from feature_engineering.py: this module is about making sure
the raw data is trustworthy; feature_engineering.py is about deriving new
signal from data that's already been validated.
"""

import pandas as pd
import numpy as np

EXPECTED_NULL_COLS = {"Churn_Date", "Churn_Reason", "Churn_Category", "Competitor_Offer"}

RANGE_CHECKS = {
    "Network_Uptime": (90.0, 100.0),
    "Customer_Satisfaction_Score": (1.0, 10.0),
    "NPS_Score": (-100, 100),
    "Support_Satisfaction": (1.0, 10.0),
    "Network_Satisfaction": (1.0, 10.0),
    "Billing_Satisfaction": (1.0, 10.0),
    "Service_Satisfaction": (1.0, 10.0),
}


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV with explicit dtypes where it matters."""
    df = pd.read_csv(path, parse_dates=["Churn_Date"])
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of null counts per column, flagging any nulls OUTSIDE
    the columns where nulls are structurally expected (e.g. Churn_Date is
    null for active customers by design, not a data quality problem).
    """
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    summary = pd.DataFrame({
        "null_count": nulls,
        "null_pct": (nulls / len(df) * 100).round(2),
        "expected": [c in EXPECTED_NULL_COLS for c in nulls.index],
    })
    unexpected = summary[~summary["expected"]]
    if len(unexpected) > 0:
        print(f"WARNING: unexpected nulls found in: {list(unexpected.index)}")
    return summary


def check_duplicates(df: pd.DataFrame, key_col: str = "Customer_ID") -> int:
    dupe_ids = df[key_col].duplicated().sum()
    dupe_rows = df.duplicated().sum()
    if dupe_ids > 0:
        print(f"WARNING: {dupe_ids} duplicate {key_col} values found")
    if dupe_rows > 0:
        print(f"WARNING: {dupe_rows} fully duplicate rows found")
    return dupe_ids


def check_value_ranges(df: pd.DataFrame) -> dict:
    """Flags any rows where a bounded metric falls outside its valid range."""
    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad) > 0:
            violations[col] = len(bad)
    return violations


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    For this dataset, nulls are structural (see EXPECTED_NULL_COLS), not
    missing-at-random — so "handling" them means making the semantics
    explicit rather than imputing values that would misrepresent an active
    customer as having a churn reason.
    """
    df = df.copy()
    df["Churn_Reason"] = df["Churn_Reason"].fillna("Not Churned")
    df["Churn_Category"] = df["Churn_Category"].fillna("Not Churned")
    df["Competitor_Offer"] = df["Competitor_Offer"].fillna("None")
    return df


def handle_duplicates(df: pd.DataFrame, key_col: str = "Customer_ID") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[key_col], keep="first")
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate {key_col} rows")
    return df


def detect_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.Series:
    """Boolean mask of IQR-based outliers for a given numeric column."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (df[col] < lower) | (df[col] > upper)


def run_full_validation(path: str) -> pd.DataFrame:
    """Convenience entry point: load, validate, clean, return a trustworthy df."""
    df = load_data(path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    check_missing_values(df)
    check_duplicates(df)
    violations = check_value_ranges(df)
    if violations:
        print(f"WARNING: range violations found: {violations}")
    else:
        print("All bounded metrics within expected ranges.")

    df = handle_missing_values(df)
    df = handle_duplicates(df)
    return df


if __name__ == "__main__":
    df = run_full_validation("/home/claude/airtel-customer-churn/data/airtel_enterprise_churn.csv")
    print(f"\nFinal validated shape: {df.shape}")
