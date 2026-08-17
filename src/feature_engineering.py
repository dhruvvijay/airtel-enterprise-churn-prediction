"""
feature_engineering.py
-----------------------
Feature engineering for the Airtel churn model. Turns the raw
airtel_enterprise_churn.csv into a model-ready feature matrix.
"""

import pandas as pd
import numpy as np

# Columns dropped before modeling: identifiers, leakage-risk churn-outcome
# fields (Churn_Date/Churn_Reason/Churn_Category only exist for churned
# customers and would leak the target), free-text/high-cardinality fields,
# and the redundant half of near-duplicate feature pairs (r > 0.9) that
# destabilize a linear model's coefficients without adding real signal:
#   Annual_Contract_Value <-> Monthly_Bill        (r = 0.999)
#   Bandwidth_Mbps <-> Monthly_Data_Usage_GB       (r = 0.967)
#   Number_of_Complaints <-> Number_of_Service_Tickets (r = 0.927)
#   Support_Response_Hours <-> Support_Resolution_Hours (r = 0.928)
# In each pair we keep the more business-meaningful / earlier-in-the-causal-
# chain feature and drop the other, rather than let the model split weight
# unstably between two columns carrying almost the same information.
DROP_COLS = ["Customer_ID", "Company_Name", "Churn_Date", "Churn_Reason",
             "Churn_Category", "Primary_Service", "Competitor_Offer",
             "Monthly_Bill", "Monthly_Data_Usage_GB",
             "Number_of_Service_Tickets", "Support_Resolution_Hours"]

CATEGORICAL_COLS = ["Company_Size", "Industry", "Customer_Type", "Customer_Segment",
                     "State", "City", "Region", "Contract_Type", "Competitor_Threat_Level"]

TARGET = "Churn"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that give the model useful non-linear signal."""
    df = df.copy()

    # Customer Lifetime Value: annual revenue x expected remaining lifetime.
    # We proxy expected lifetime with tenure-to-date plus a modest forward
    # horizon that shrinks as satisfaction drops (a rough, explainable proxy).
    expected_forward_years = np.clip(df["Customer_Satisfaction_Score"] / 2, 0.5, 5)
    df["CLV"] = df["Annual_Contract_Value"] * (df["Years_With_Airtel"] + expected_forward_years) / \
        (df["Years_With_Airtel"] + 1)

    # Support burden ratio: tickets per year of tenure (spikes flag recent deterioration)
    df["Tickets_per_Tenure_Year"] = df["Number_of_Service_Tickets"] / (df["Years_With_Airtel"] + 0.1)

    # Complaint-to-satisfaction mismatch: complaints despite claiming satisfaction (inconsistency signal)
    df["Complaint_Satisfaction_Gap"] = df["Number_of_Complaints"] * (10 - df["Customer_Satisfaction_Score"])

    # Contract risk window: contract ending soon = renewal decision point
    df["Contract_Ending_Soon"] = (df["Contract_Remaining_Months"] <= 3).astype(int)

    # Price pressure: only meaningful when a competitor was actually considered
    df["Price_Pressure"] = np.where(
        df["Competitor_Considered"] == 1,
        -df["Competitor_Price_Difference"],  # more positive = competitor more attractively priced
        0
    )

    # Revenue-weighted quality gap: high-value customers experiencing poor quality
    df["Revenue_Weighted_Downtime"] = df["Downtime_Hours"] * df["Annual_Contract_Value"] / 1_000_000

    # Service breadth relative to company size tier (SMB with many services = sticky; Enterprise with few = risk)
    size_map = {"Small": 0, "Medium": 1, "Large": 2, "Enterprise": 3}
    df["Company_Size_Rank"] = df["Company_Size"].map(size_map)
    df["Services_per_Size_Tier"] = df["Number_of_Services"] / (df["Company_Size_Rank"] + 1)

    return df


def prepare_model_matrix(df: pd.DataFrame, fit_encoder=None):
    """
    Returns (X, y, encoder_categories) ready for sklearn.
    If fit_encoder is provided (dict of {col: categories}), reuses those
    categories for one-hot encoding so train/serve stay consistent.
    """
    df = engineer_features(df)
    y = df[TARGET] if TARGET in df.columns else None

    feature_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET],
                          errors="ignore")

    if fit_encoder is None:
        feature_df = pd.get_dummies(feature_df, columns=CATEGORICAL_COLS, drop_first=False)
        encoder_categories = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_COLS}
    else:
        for c in CATEGORICAL_COLS:
            feature_df[c] = pd.Categorical(feature_df[c], categories=fit_encoder[c])
        feature_df = pd.get_dummies(feature_df, columns=CATEGORICAL_COLS, drop_first=False)
        encoder_categories = fit_encoder

    return feature_df, y, encoder_categories
