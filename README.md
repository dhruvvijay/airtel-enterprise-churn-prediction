# Airtel Enterprise Customer Churn Prediction & Service Intelligence Dashboard

> **Synthetic data disclaimer:** This project uses a synthetically generated enterprise customer dataset inspired by publicly available Airtel Business service categories (connectivity, cloud, security, IoT, SD-WAN, managed services, voice/collaboration). It does **not** use, reference, or imply access to real Airtel customer data. All company names, individual records, and churn events are fictional and generated for analytical/educational purposes.

A complete, end-to-end churn analytics and prediction system built on a 25,000-record synthetic enterprise telecom dataset: SQL analysis, exploratory data analysis, a machine-learning churn model with explainability, a retention recommendation engine, a Power BI dashboard design, and a live Streamlit predictor.

## Project Overview

Enterprise B2B telecom churn is expensive and hard to see coming — by the time a large account cancels, the warning signs (rising downtime, slower support, mounting complaints) were usually visible months earlier. This project builds a full pipeline to surface those signals: from raw service-quality data through a trained probability model to a prioritized, revenue-weighted retention list a real account team could act on.

## Business Problem

Given enterprise customer data spanning services, geography, support interactions, and satisfaction, answer:
- Which customers are likely to churn, and why?
- Which services, cities, states, and industries are underperforming?
- How much revenue is actually at risk?
- Who should the retention team call first, and what should they say?

## Objectives

1. Quantify churn drivers with real statistical evidence (not assumptions)
2. Distinguish churn *rate* problems (service quality issues) from churn *volume* problems (large-market exposure)
3. Build a probability-based churn model that beats a naive baseline on the metrics that matter (recall, ROC-AUC — not just accuracy)
4. Turn model output into a prioritized, explainable, actionable retention list
5. Package all of the above into portfolio-ready deliverables: notebooks, SQL, a live dashboard, and a BI tool spec

## Dataset

**Synthetic** — 25,000 enterprise customer records, 69 columns, spanning 18 states, 29 cities, and 17 industries. Churn (21.43% base rate) is generated from a weighted logistic function of real service-quality, support, and competitive-pressure signals (not randomly assigned), with noise added so the resulting ML problem is realistic rather than trivially separable. Full column reference: [`reports/DATA_DICTIONARY.md`](reports/DATA_DICTIONARY.md). Generation methodology: [`src/generate_dataset.py`](src/generate_dataset.py).

## Technologies Used

- **Python**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib
- **Database**: MySQL (schema + 15+ analysis queries using CTEs, window functions, JOINs, CASE)
- **ML**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (XGBoost substituted with sklearn's GradientBoostingClassifier — not installable in the build environment, per the project's own fallback plan)
- **Dashboards**: Power BI (design spec + DAX), Streamlit + Plotly (live app)
- **Tooling**: Jupyter Notebook, Git

## Architecture

```
Synthetic Customer Data (Python/NumPy, weighted logistic churn model)
        ↓
Python / Pandas — cleaning, validation
        ↓
EDA (01_EDA_Airtel_Churn.ipynb)
        ↓
MySQL — normalized schema (customers, service_catalog, customer_services bridge)
        ↓
Feature Engineering (CLV, complaint-satisfaction gap, price pressure, etc.)
        ↓
Machine Learning (02_Churn_Prediction_Model.ipynb) — 4 models compared
        ↓
Churn Probability → Risk Tiers → Retention Priority Score
        ↓
Power BI (design spec) / Streamlit (live app)
        ↓
Business Recommendations (reports/business_insights.md)
```

## Project Structure

```text
airtel-customer-churn/
├── data/airtel_enterprise_churn.csv
├── notebooks/
│   ├── 01_EDA_Airtel_Churn.ipynb
│   └── 02_Churn_Prediction_Model.ipynb
├── sql/
│   ├── 00_schema_and_load.sql
│   ├── airtel_churn_analysis.sql
│   ├── service_catalog.csv
│   └── customer_services.csv
├── dashboard/streamlit_app.py
├── models/
│   ├── churn_model.pkl
│   ├── churn_predictions.csv
│   └── model_comparison.json
├── src/
│   ├── generate_dataset.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── prediction.py
│   └── compute_insights.py
├── reports/
│   ├── DATA_DICTIONARY.md
│   ├── business_insights.md
│   ├── powerbi_dashboard_design.md
│   ├── resume_bullets.md
│   └── interview_prep.md
├── requirements.txt
├── README.md
└── LICENSE
```

## Key Insights

*(Full write-up with methodology: [`reports/business_insights.md`](reports/business_insights.md). Every number below is computed directly from the dataset, not illustrative.)*

- **21.43% overall churn rate** (5,358 of 25,000 customers); **₹1,405 Cr** of annual contract value already lost, **17.5%** of total portfolio value
- **Frequent Downtime is the #1 reason customers leave** — 33.97% of all churn, more than the next two reasons combined
- **Downtime hours (+0.40), outage count (+0.38), and network uptime (-0.38) are the strongest correlates of churn** — stronger than price, contract type, or even raw satisfaction score
- Churn **rate** is highest in Goa (25.1%) and Andhra Pradesh (23.7%), but churn **volume** is concentrated in Maharashtra (895 customers) and Karnataka (653) — the two rankings answer different business questions and both matter
- **Large Enterprise customers are 20% of the base but 58% of portfolio value**, and churn *less* than average (17.9% vs 21.4%) — SMB customers churn most (23.7%) but represent the smallest revenue share
- Contract length (multi-year vs monthly) shows almost no protective effect against churn — **service quality is the retention lever that actually works here, not lock-in**

## Machine Learning Results

Four models trained and compared; XGBoost substituted with GradientBoostingClassifier (not installable in this environment).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** (selected) | 0.766 | 0.472 | **0.775** | 0.587 | **0.848** |
| Gradient Boosting | 0.829 | 0.659 | 0.418 | 0.511 | 0.842 |
| Random Forest | 0.825 | 0.608 | 0.518 | 0.559 | 0.840 |
| Decision Tree | 0.750 | 0.448 | 0.716 | 0.551 | 0.801 |

**Why Logistic Regression, despite lower accuracy:** churn prediction is imbalanced (~21% positive class), and for this business problem the cost of *missing* an at-risk enterprise account generally exceeds the cost of one unnecessary retention call. Logistic Regression has the highest ROC-AUC and by far the highest recall — it catches 77.5% of actual churners, vs. 41.8% for Gradient Boosting. A majority-class baseline would hit 78.6% accuracy while catching zero churners, which is exactly why accuracy alone is the wrong metric to optimize here.

**Top predictive features:** Customer Satisfaction Score, Downtime Hours, Number of Complaints, SLA Breaches, Network Uptime, NPS Score — service-quality and experience signals dominate, not commercial/size fields. (An earlier version of this model had `Monthly_Bill`/`Annual_Contract_Value` as the top two "drivers" purely due to near-perfect collinearity between those two columns — caught via correlation analysis and fixed by dropping the redundant feature. See `notebooks/02_Churn_Prediction_Model.ipynb` §3.)

## Dashboards

- **Streamlit** (`dashboard/streamlit_app.py`): 4 tabs — Overview, Geography & Services, Risk & Retention, and a live churn predictor. Run with `streamlit run dashboard/streamlit_app.py`.
- **Power BI**: full data model, DAX measures, and 5-page design spec in [`reports/powerbi_dashboard_design.md`](reports/powerbi_dashboard_design.md) (binary `.pbix` can't be generated outside Power BI Desktop — this doc is the exact blueprint to build it).

*Dashboard screenshots: add after running the Streamlit app / building the Power BI file locally.*

## Business Recommendations

1. Prioritize downtime reduction for high-revenue accounts first (revenue-weighted downtime is the single largest lever)
2. Fix support *response* time specifically, not just resolution time
3. Flag accounts with 2+ SLA breaches for proactive account management — churn rate step-changes at that threshold
4. Don't rely on longer contracts as a retention strategy — the data shows minimal protective effect
5. Watch Energy and Financial Services accounts more closely (highest industry churn rates)

Full detail and supporting numbers: [`reports/business_insights.md`](reports/business_insights.md).

## Future Improvements

- Swap in XGBoost/LightGBM and true SHAP values once available (the model comparison and explainability layer are both structured to make this a drop-in swap — see the fallback notes in `src/train_model.py` and `src/prediction.py`)
- Add real time-series churn tracking (the synthetic `Churn_Date` field supports this, not yet visualized as a trend)
- A/B test retention interventions against the model's recommendations to validate the what-if scenarios in `src/prediction.py`
- Replace the synthetic dataset with anonymized real data if this pattern is ever applied to an actual telecom dataset

## Setup

```bash
pip install -r requirements.txt
python src/generate_dataset.py 25000 data/airtel_enterprise_churn.csv   # regenerate data (optional, already included)
python src/train_model.py                                                # train + save model
python src/prediction.py                                                 # score all customers, generate risk list
streamlit run dashboard/streamlit_app.py                                 # launch live dashboard
```

For MySQL: run `sql/00_schema_and_load.sql` then `sql/airtel_churn_analysis.sql` in MySQL Workbench or the CLI (adjust file paths for `LOAD DATA LOCAL INFILE`).
