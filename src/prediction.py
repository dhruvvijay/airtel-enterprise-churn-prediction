"""
prediction.py
--------------
Churn prediction system: scores every customer, assigns risk tiers,
computes revenue-at-risk and CLV-based retention priority, generates a
per-customer explanation of WHY they're at risk (a lightweight stand-in
for SHAP, since the `shap` package isn't installable in this environment),
and applies a rule-based retention recommendation engine.

Outputs:
    models/churn_predictions.csv  -> loaded into MySQL's churn_predictions table
"""

import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, '/home/claude/airtel-customer-churn/src')
from feature_engineering import prepare_model_matrix, engineer_features
from data_preprocessing import run_full_validation

MODEL_PATH = "/home/claude/airtel-customer-churn/models/churn_model.pkl"
DATA_PATH = "/home/claude/airtel-customer-churn/data/airtel_enterprise_churn.csv"
OUT_PATH = "/home/claude/airtel-customer-churn/models/churn_predictions.csv"


def risk_category(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    elif prob < 0.80:
        return "High"
    else:
        return "Critical"


def retention_priority(prob, revenue, clv, num_services, contract_remaining):
    """
    Retention_Priority_Score combines churn probability, revenue value,
    CLV, service breadth, and contract urgency into one 0-100 score, then
    buckets it into Critical / High / Medium / Low for the retention team.
    """
    # normalize revenue & CLV on a log scale so a few huge accounts don't
    # completely dominate the score for mid-market customers
    revenue_score = np.clip(np.log1p(revenue) / np.log1p(revenue.max()), 0, 1)
    clv_score = np.clip(np.log1p(clv) / np.log1p(clv.max()), 0, 1)
    service_score = np.clip(num_services / num_services.max(), 0, 1)
    urgency_score = np.where(contract_remaining <= 3, 1.0,
                     np.where(contract_remaining <= 6, 0.6, 0.2))

    score = (
        0.45 * prob +
        0.25 * revenue_score +
        0.15 * clv_score +
        0.05 * service_score +
        0.10 * urgency_score
    ) * 100
    return np.round(score, 1)


def priority_bucket(score, prob, revenue_score):
    if prob >= 0.60 and revenue_score >= 0.7:
        return "Critical"
    elif prob >= 0.60 and revenue_score >= 0.4:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Lightweight local explainability (SHAP is not installable in this
# environment; this approximates "why did this customer score the way they
# did" using the model's own linear weights x this customer's standardized
# deviation from the population mean for each feature -- for a logistic
# model this is mathematically the exact per-feature logit contribution;
# for tree models it degrades gracefully to importance x deviation).
# ---------------------------------------------------------------------------
def explain_customer(row_scaled_or_raw, feature_names, model, uses_scaling, feature_means, feature_stds):
    if hasattr(model, "coef_"):
        contributions = model.coef_[0] * row_scaled_or_raw
    else:
        # tree-based fallback: importance-weighted standardized deviation
        importances = model.feature_importances_
        z = (row_scaled_or_raw - feature_means) / (feature_stds + 1e-9)
        contributions = importances * z

    contrib_series = pd.Series(contributions, index=feature_names)
    top_positive = contrib_series.sort_values(ascending=False).head(5)
    # keep only features that actually push risk UP (positive contribution)
    top_risk_factors = [f for f in top_positive.index if top_positive[f] > 0][:3]
    if not top_risk_factors:
        top_risk_factors = ["No dominant single risk factor — profile is broadly healthy"]
    return top_risk_factors


READABLE_NAMES = {
    "Downtime_Hours": "High network downtime",
    "Number_of_Outages": "Frequent service outages",
    "SLA_Breaches": "Repeated SLA breaches",
    "Support_Response_Hours": "Slow support response time",
    "Support_Resolution_Hours": "Slow issue resolution",
    "Number_of_Complaints": "High complaint volume",
    "Number_of_Service_Tickets": "High support ticket volume",
    "Billing_Issues": "Recurring billing issues",
    "Packet_Loss_Percentage": "High packet loss",
    "Average_Latency_ms": "High network latency",
    "Network_Uptime": "Below-average network uptime",
    "Customer_Satisfaction_Score": "Low overall satisfaction",
    "NPS_Score": "Low NPS / low willingness to recommend",
    "Network_Satisfaction": "Low network satisfaction",
    "Support_Satisfaction": "Low support satisfaction",
    "Service_Quality_Score": "Below-average service quality",
    "Complaint_Satisfaction_Gap": "Complaints despite prior loyalty",
    "Revenue_Weighted_Downtime": "High-value account with high downtime",
    "Tickets_per_Tenure_Year": "Rising ticket rate relative to tenure",
    "Price_Pressure": "Competitor offering a materially lower price",
    "Contract_Ending_Soon": "Contract renewal window approaching",
}


def readable(factor):
    return READABLE_NAMES.get(factor, factor.replace("_", " "))


# ---------------------------------------------------------------------------
# Retention recommendation rule engine
# ---------------------------------------------------------------------------
def recommend_action(row, risk_cat):
    if risk_cat == "Low":
        return "No action needed — monitor at standard cadence"

    high_downtime = row["Downtime_Hours"] > 40
    high_revenue = row["Annual_Contract_Value"] > row["_acv_p75"]
    price_sensitive_with_competitor = row["Competitor_Considered"] == 1 and row["Competitor_Price_Difference"] < -10
    poor_support_sat = row["Support_Satisfaction"] < 5
    multi_sla_breach = row["SLA_Breaches"] >= 2
    low_usage = row["Number_of_Services"] <= 1
    contract_ending_soon = row["Contract_Remaining_Months"] <= 3

    actions = []
    if high_downtime and high_revenue:
        actions.append("Priority network-quality intervention (dedicated NOC escalation)")
    if price_sensitive_with_competitor:
        actions.append("Pricing / contract review with retention desk")
    if poor_support_sat:
        actions.append("Assign dedicated account manager")
    if multi_sla_breach:
        actions.append("Technical escalation + SLA credit review")
    if low_usage:
        actions.append("Product adoption / cross-sell campaign")
    if contract_ending_soon and risk_cat in ("High", "Critical"):
        actions.append("Proactive renewal outreach with retention offer")

    if not actions:
        actions.append("Standard retention call — reassess satisfaction drivers")

    return "; ".join(actions[:2])  # cap at top 2 actions to keep it actionable


def main():
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    encoder_categories = artifact["encoder_categories"]
    uses_scaling = artifact["uses_scaling"]

    df = run_full_validation(DATA_PATH)
    X, y, _ = prepare_model_matrix(df, fit_encoder=encoder_categories)
    X = X.reindex(columns=feature_names, fill_value=0).astype(np.float64)

    X_model = scaler.transform(X) if uses_scaling else X.values
    churn_probability = model.predict_proba(X_model)[:, 1]

    df["Churn_Probability"] = np.round(churn_probability, 4)
    df["Risk_Category"] = df["Churn_Probability"].apply(risk_category)
    df["Revenue_at_Risk"] = np.round(df["Annual_Contract_Value"] * df["Churn_Probability"], 2)

    # CLV (re-derive from engineered features to keep this file self-contained)
    engineered = engineer_features(df)
    df["CLV"] = np.round(engineered["CLV"], 2)

    df["Retention_Priority_Score"] = retention_priority(
        df["Churn_Probability"].values, df["Annual_Contract_Value"].values,
        df["CLV"].values, df["Number_of_Services"].values, df["Contract_Remaining_Months"].values
    )
    revenue_pct = df["Annual_Contract_Value"].rank(pct=True).values
    df["Retention_Priority_Bucket"] = [
        priority_bucket(s, p, r) for s, p, r in
        zip(df["Retention_Priority_Score"], df["Churn_Probability"], revenue_pct)
    ]

    # ---- per-customer explanation ----
    feature_means = X.values.mean(axis=0)
    feature_stds = X.values.std(axis=0)
    top_factors_list = []
    for i in range(len(df)):
        row_vec = X_model[i] if uses_scaling else (X.values[i] - feature_means) / (feature_stds + 1e-9)
        factors = explain_customer(row_vec, feature_names, model, uses_scaling, feature_means, feature_stds)
        top_factors_list.append(factors)

    df["Top_Risk_Factors"] = [
        ", ".join(readable(f) for f in factors) for factors in top_factors_list
    ]
    df["Main_Risk_Factor"] = [
        readable(factors[0]) if factors else "N/A" for factors in top_factors_list
    ]

    # ---- retention recommendation ----
    df["_acv_p75"] = df["Annual_Contract_Value"].quantile(0.75)
    df["Recommended_Action"] = df.apply(lambda r: recommend_action(r, r["Risk_Category"]), axis=1)
    df.drop(columns=["_acv_p75"], inplace=True)

    # ---- export for SQL / Streamlit / README ----
    export_cols = ["Customer_ID", "Company_Name", "Industry", "State", "City",
                    "Annual_Contract_Value", "Churn_Probability", "Risk_Category",
                    "Revenue_at_Risk", "CLV", "Retention_Priority_Score",
                    "Retention_Priority_Bucket", "Main_Risk_Factor", "Top_Risk_Factors",
                    "Recommended_Action"]
    export_df = df[export_cols].copy()
    # match the churn_predictions SQL table's exact columns (subset)
    sql_export = export_df.rename(columns={
        "Main_Risk_Factor": "Main_Risk_Factor"
    })[["Customer_ID", "Churn_Probability", "Risk_Category", "Revenue_at_Risk",
        "Main_Risk_Factor", "Recommended_Action"]]
    sql_export.to_csv("/home/claude/airtel-customer-churn/models/churn_predictions_sql.csv", index=False)
    export_df.to_csv(OUT_PATH, index=False)

    print(f"Scored {len(df):,} customers -> {OUT_PATH}")
    print(f"\nRisk distribution:")
    print(df["Risk_Category"].value_counts())
    print(f"\nRetention priority distribution:")
    print(df["Retention_Priority_Bucket"].value_counts())
    print(f"\nTotal revenue at risk (probability-weighted): Rs {df['Revenue_at_Risk'].sum():,.0f}")

    critical = df[df["Retention_Priority_Bucket"] == "Critical"].sort_values(
        "Retention_Priority_Score", ascending=False).head(10)
    print(f"\nTop 10 Critical-priority customers:")
    print(critical[["Company_Name", "Churn_Probability", "Annual_Contract_Value",
                     "Main_Risk_Factor", "Recommended_Action"]].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# What-if scenario analysis
# ---------------------------------------------------------------------------
def what_if(df_original, scenario: str, magnitude: float = 0.2):
    """
    Re-scores the population under a simple stated scenario. These are
    scenario ESTIMATES from the trained model, not guaranteed outcomes.
    Supported scenarios: 'reduce_downtime', 'improve_support_response',
    'retention_offer_high_risk'.
    """
    artifact = joblib.load(MODEL_PATH)
    model, scaler = artifact["model"], artifact["scaler"]
    feature_names, encoder_categories = artifact["feature_names"], artifact["encoder_categories"]
    uses_scaling = artifact["uses_scaling"]

    df = df_original.copy()
    if scenario == "reduce_downtime":
        df["Downtime_Hours"] = df["Downtime_Hours"] * (1 - magnitude)
        df["Number_of_Outages"] = (df["Number_of_Outages"] * (1 - magnitude)).round()
    elif scenario == "improve_support_response":
        df["Support_Response_Hours"] = df["Support_Response_Hours"] * (1 - magnitude)
        df["Support_Resolution_Hours"] = df["Support_Resolution_Hours"] * (1 - magnitude)
    elif scenario == "retention_offer_high_risk":
        # simulate a modest satisfaction + price-pressure relief for high-risk accounts
        mask = df.get("Risk_Category", pd.Series(index=df.index)).isin(["High", "Critical"]) \
            if "Risk_Category" in df.columns else pd.Series(False, index=df.index)
        df.loc[mask, "Customer_Satisfaction_Score"] = np.clip(
            df.loc[mask, "Customer_Satisfaction_Score"] * (1 + magnitude), 1, 10)
        df.loc[mask, "Competitor_Price_Difference"] = df.loc[mask, "Competitor_Price_Difference"] + magnitude * 10

    X, _, _ = prepare_model_matrix(df, fit_encoder=encoder_categories)
    X = X.reindex(columns=feature_names, fill_value=0).astype(np.float64)
    X_model = scaler.transform(X) if uses_scaling else X.values
    new_prob = model.predict_proba(X_model)[:, 1]

    before = df_original.get("Churn_Probability")
    result = {
        "scenario": scenario,
        "magnitude": magnitude,
        "avg_churn_probability_before": round(float(before.mean()), 4) if before is not None else None,
        "avg_churn_probability_after": round(float(new_prob.mean()), 4),
        "note": "Scenario ESTIMATE from the trained model, not a guaranteed business outcome.",
    }
    return result


if __name__ == "__main__":
    main()
