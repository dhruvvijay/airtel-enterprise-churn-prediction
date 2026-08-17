"""
streamlit_app.py
-----------------
Airtel Enterprise Customer Churn & Service Intelligence — Streamlit dashboard.

Run with:  streamlit run streamlit_app.py
(from inside the dashboard/ folder, with ../data and ../models available)
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import prepare_model_matrix

st.set_page_config(page_title="Airtel Enterprise Churn Intelligence", layout="wide", page_icon="📡")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "airtel_enterprise_churn.csv.gz")
PRED_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_predictions.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")

RED = "#E60000"
DARK = "#1A1A1A"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    try:
        preds = pd.read_csv(PRED_PATH)
        df = df.merge(
            preds[["Customer_ID", "Churn_Probability", "Risk_Category", "Revenue_at_Risk",
                   "CLV", "Retention_Priority_Score", "Retention_Priority_Bucket",
                   "Main_Risk_Factor", "Recommended_Action"]],
            on="Customer_ID", how="left"
        )
    except FileNotFoundError:
        st.warning("Run `python src/prediction.py` first to generate risk scores.")
    return df


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


df = load_data()
model_artifact = load_model()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background-color:{RED};padding:18px 24px;border-radius:8px;margin-bottom:20px;">
        <h1 style="color:white;margin:0;">📡 Airtel Enterprise Customer Churn & Service Intelligence</h1>
        <p style="color:white;margin:4px 0 0 0;font-size:14px;">
            Synthetic dataset for portfolio/educational purposes — not real Airtel customer data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

states = st.sidebar.multiselect("State", sorted(df["State"].unique()))
cities = st.sidebar.multiselect("City", sorted(df["City"].unique()))
industries = st.sidebar.multiselect("Industry", sorted(df["Industry"].unique()))
segments = st.sidebar.multiselect("Customer Segment", sorted(df["Customer_Segment"].unique()))
risk_levels = st.sidebar.multiselect("Risk Level", ["Low", "Medium", "High", "Critical"])

service_cols = ["Internet_Leased_Line", "Dedicated_Internet", "MPLS_VPN", "SD_WAN", "Broadband",
                 "Airtel_Cloud", "Multi_Cloud_Connect", "Managed_Firewall", "Business_Voice", "IoT_Services"]
service_filter = st.sidebar.multiselect("Service Subscribed", service_cols)

filtered = df.copy()
if states:
    filtered = filtered[filtered["State"].isin(states)]
if cities:
    filtered = filtered[filtered["City"].isin(cities)]
if industries:
    filtered = filtered[filtered["Industry"].isin(industries)]
if segments:
    filtered = filtered[filtered["Customer_Segment"].isin(segments)]
if risk_levels and "Risk_Category" in filtered.columns:
    filtered = filtered[filtered["Risk_Category"].isin(risk_levels)]
for svc in service_filter:
    filtered = filtered[filtered[svc] == 1]

st.sidebar.markdown(f"**{len(filtered):,}** customers match current filters")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🌍 Geography & Services", "⚠️ Risk & Retention", "🔮 Live Predictor"])

# ---------------------------------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    total = len(filtered)
    churned = int(filtered["Churn"].sum())
    churn_rate = (churned / total * 100) if total else 0
    revenue_at_risk = filtered.loc[filtered["Churn"] == 1, "Annual_Contract_Value"].sum()

    c1.metric("Customers", f"{total:,}")
    c2.metric("Churned", f"{churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Revenue at Risk (churned ACV)", f"₹{revenue_at_risk/1e7:,.1f} Cr")

    col1, col2 = st.columns(2)
    with col1:
        reason_df = filtered[filtered["Churn"] == 1]["Churn_Reason"].value_counts().reset_index()
        reason_df.columns = ["Reason", "Customers"]
        fig = px.bar(reason_df, x="Customers", y="Reason", orientation="h",
                     title="Churn Reasons", color_discrete_sequence=[RED])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_df = filtered.groupby("Customer_Segment")["Churn"].mean().reset_index()
        seg_df["Churn"] = seg_df["Churn"] * 100
        fig = px.bar(seg_df, x="Customer_Segment", y="Churn",
                     title="Churn Rate by Customer Segment (%)", color_discrete_sequence=[RED])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        ind_df = filtered.groupby("Industry")["Churn"].mean().sort_values(ascending=False).head(10).reset_index()
        ind_df["Churn"] = ind_df["Churn"] * 100
        fig = px.bar(ind_df, x="Churn", y="Industry", orientation="h",
                     title="Top 10 Industries by Churn Rate (%)", color_discrete_sequence=[RED])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.pie(filtered, names=filtered["Churn"].map({0: "Active", 1: "Churned"}),
                     title="Active vs Churned", color_discrete_sequence=["#2ca02c", RED])
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Geography & Services
# ---------------------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        state_df = filtered.groupby("State").agg(Total=("Customer_ID", "count"), Churned=("Churn", "sum")).reset_index()
        state_df["Churn_Rate"] = (state_df["Churned"] / state_df["Total"] * 100).round(1)
        fig = px.bar(state_df.sort_values("Churn_Rate", ascending=False).head(10),
                     x="Churn_Rate", y="State", orientation="h",
                     title="Top 10 States by Churn Rate (%)", color_discrete_sequence=[RED])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_df = filtered.groupby("City").agg(Total=("Customer_ID", "count"), Churned=("Churn", "sum")).reset_index()
        fig = px.bar(city_df.sort_values("Churned", ascending=False).head(10),
                     x="Churned", y="City", orientation="h",
                     title="Top 10 Cities by Churned Volume", color_discrete_sequence=[RED])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Service Churn Rate")
    rows = []
    for s in service_cols:
        sub = filtered[filtered[s] == 1]
        if len(sub):
            rows.append({"Service": s, "Customers": len(sub), "Churn_Rate": round(sub["Churn"].mean() * 100, 1)})
    svc_df = pd.DataFrame(rows).sort_values("Churn_Rate", ascending=False)
    fig = px.bar(svc_df, x="Churn_Rate", y="Service", orientation="h",
                 title="Churn Rate by Service (%)", color_discrete_sequence=[RED])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Risk & Retention
# ---------------------------------------------------------------------------
with tab3:
    if "Risk_Category" not in filtered.columns:
        st.warning("Run `python src/prediction.py` to generate risk scores first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High Risk", f"{(filtered['Risk_Category']=='High').sum():,}")
        c2.metric("Critical Risk", f"{(filtered['Risk_Category']=='Critical').sum():,}")
        c3.metric("Avg Churn Probability", f"{filtered['Churn_Probability'].mean()*100:.1f}%")
        c4.metric("Total Predicted Revenue at Risk", f"₹{filtered['Revenue_at_Risk'].sum()/1e7:,.1f} Cr")

        risk_dist = filtered["Risk_Category"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).reset_index()
        risk_dist.columns = ["Risk", "Customers"]
        fig = px.bar(risk_dist, x="Risk", y="Customers", title="Risk Distribution",
                     color="Risk", color_discrete_map={"Low": "#2ca02c", "Medium": "#f1c40f",
                                                          "High": "#e67e22", "Critical": RED})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Priority Retention List — Active customers, highest priority first")
        priority = filtered[filtered["Churn"] == 0].sort_values("Retention_Priority_Score", ascending=False).head(25)
        st.dataframe(
            priority[["Company_Name", "Industry", "State", "Annual_Contract_Value", "Churn_Probability",
                      "Risk_Category", "Retention_Priority_Bucket", "Main_Risk_Factor", "Recommended_Action"]],
            use_container_width=True, hide_index=True
        )

# ---------------------------------------------------------------------------
# TAB 4 — Live Predictor
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Predict Churn Probability for a New / Hypothetical Customer")

    if model_artifact is None:
        st.error("Model not found. Run `python src/train_model.py` first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Years with Airtel", 0.1, 15.0, 3.0)
            monthly_bill = st.number_input("Monthly Bill (₹)", 5000, 2000000, 80000, step=5000)
            uptime = st.slider("Network Uptime (%)", 90.0, 100.0, 98.5)
            downtime = st.slider("Downtime Hours", 0.0, 200.0, 25.0)
        with col2:
            complaints = st.slider("Number of Complaints", 0, 30, 2)
            satisfaction = st.slider("Customer Satisfaction (1-10)", 1.0, 10.0, 7.5)
            support_response = st.slider("Support Response Time (hours)", 0.0, 72.0, 6.0)
            sla_breaches = st.slider("SLA Breaches", 0, 6, 0)
        with col3:
            num_services = st.slider("Number of Services", 1, 15, 4)
            contract_remaining = st.slider("Contract Remaining (months)", 0, 36, 12)
            company_size = st.selectbox("Company Size", ["Small", "Medium", "Large", "Enterprise"])
            competitor = st.selectbox("Considering a competitor?", ["No", "Yes"])

        if st.button("Predict Churn Probability", type="primary"):
            row = {c: 0 for c in df.columns if c not in
                   ["Customer_ID", "Company_Name", "Churn", "Churn_Date", "Churn_Reason", "Churn_Category"]}
            row.update({
                "Years_With_Airtel": tenure, "Monthly_Bill": monthly_bill,
                "Annual_Contract_Value": monthly_bill * 12,
                "Network_Uptime": uptime, "Downtime_Hours": downtime,
                "Number_of_Complaints": complaints, "Customer_Satisfaction_Score": satisfaction,
                "Support_Response_Hours": support_response, "SLA_Breaches": sla_breaches,
                "Number_of_Services": num_services, "Contract_Remaining_Months": contract_remaining,
                "Company_Size": company_size,
                "Competitor_Considered": 1 if competitor == "Yes" else 0,
                "Competitor_Price_Difference": -12.0 if competitor == "Yes" else 0.0,
                "Industry": "IT", "Customer_Type": "Existing Business", "Customer_Segment": "Mid-Market",
                "State": "Maharashtra", "City": "Mumbai", "Region": "West", "Contract_Type": "Annual",
                "Competitor_Threat_Level": "Medium" if competitor == "Yes" else "Low",
                "Support_Satisfaction": satisfaction, "Network_Satisfaction": satisfaction,
                "Billing_Satisfaction": satisfaction, "Service_Satisfaction": satisfaction,
                "NPS_Score": int((satisfaction - 5.5) * 20),
                "Service_Quality_Score": uptime * 0.6 + (100 - support_response) * 0.4,
                "Number_of_Service_Tickets": complaints + 1, "Billing_Issues": 0,
                "Average_Latency_ms": 20.0, "Packet_Loss_Percentage": 0.5,
                "Support_Resolution_Hours": support_response * 3, "Number_of_Outages": max(1, int(downtime / 8)),
                "Installation_Delay_Days": 3, "Bandwidth_Mbps": 200, "Monthly_Data_Usage_GB": 40000,
                "Service_Tenure": tenure, "Number_of_IoT_Devices": 0, "Pincode_Zone": 400001,
                "Competitor_Service_Rating": 7.0 if competitor == "Yes" else 0.0,
            })
            input_df = pd.DataFrame([row])

            X, _, _ = prepare_model_matrix(input_df, fit_encoder=model_artifact["encoder_categories"])
            X = X.reindex(columns=model_artifact["feature_names"], fill_value=0).astype(np.float64)
            X_model = model_artifact["scaler"].transform(X) if model_artifact["uses_scaling"] else X.values
            prob = model_artifact["model"].predict_proba(X_model)[0, 1]

            risk = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH" if prob < 0.8 else "CRITICAL"
            color = {"LOW": "#2ca02c", "MEDIUM": "#f1c40f", "HIGH": "#e67e22", "CRITICAL": RED}[risk]

            st.markdown(f"### Churn Probability: **{prob*100:.1f}%**")
            st.markdown(f"### Risk Level: <span style='color:{color}'>**{risk}**</span>", unsafe_allow_html=True)

            reasons = []
            if downtime > 30: reasons.append("high downtime")
            if complaints > 5: reasons.append("high complaint volume")
            if satisfaction < 6: reasons.append("low satisfaction")
            if sla_breaches > 1: reasons.append("repeated SLA breaches")
            if support_response > 12: reasons.append("slow support response")
            if competitor == "Yes": reasons.append("actively considering a competitor")
            if not reasons: reasons.append("no major single risk factor — profile looks healthy")
            st.markdown(f"**Main Risk Factors:** {', '.join(reasons)}")

            if risk in ("HIGH", "CRITICAL"):
                st.markdown("**Recommended Action:** Immediate account intervention — priority network/support escalation and retention outreach.")
            elif risk == "MEDIUM":
                st.markdown("**Recommended Action:** Proactive check-in and satisfaction review.")
            else:
                st.markdown("**Recommended Action:** Standard monitoring cadence.")

st.markdown("---")
st.caption("Synthetic data project for portfolio/educational purposes. Not affiliated with or using real Airtel customer data.")
