# Power BI Dashboard — Data Model & Design Spec

Power BI Desktop (.pbix) is a binary file that can't be produced directly in this environment, so this document is the exact blueprint to build it: data model, relationships, DAX measures, and page-by-page layout. Following this should take ~1-2 hours in Power BI Desktop.

## 1. Data Model

**Import these tables** (Get Data → Text/CSV):
1. `airtel_enterprise_churn.csv` → table name `Customers`
2. `sql/customer_services.csv` → table name `CustomerServices`
3. `sql/service_catalog.csv` → table name `ServiceCatalog`
4. `models/churn_predictions.csv` → table name `ChurnPredictions`

**Create a Date table** (for time-intelligence on `Churn_Date`):
```dax
DateTable = CALENDAR(DATE(2024,8,1), DATE(2026,8,17))
```
Mark it as a Date Table (Model view → right-click → Mark as Date Table), and add:
```dax
Month_Year = FORMAT('DateTable'[Date], "MMM YYYY")
```

**Relationships** (Model view):
- `Customers[Customer_ID]` (1) → `CustomerServices[Customer_ID]` (many)
- `ServiceCatalog[Service_Name]` (1) → `CustomerServices[Service_Name]` (many)
- `Customers[Customer_ID]` (1) → `ChurnPredictions[Customer_ID]` (1, one-to-one)
- `DateTable[Date]` (1) → `Customers[Churn_Date]` (many) — set to **inactive**, activate only in visuals that need churn-date trending via `USERELATIONSHIP()`

## 2. Key DAX Measures

```dax
Total Customers = COUNTROWS(Customers)

Churned Customers = CALCULATE(COUNTROWS(Customers), Customers[Churn] = 1)

Active Customers = CALCULATE(COUNTROWS(Customers), Customers[Churn] = 0)

Churn Rate = DIVIDE([Churned Customers], [Total Customers], 0)

Revenue at Risk (Actual) =
    CALCULATE(SUM(Customers[Annual_Contract_Value]), Customers[Churn] = 1)

Revenue at Risk (Predicted) = SUM(ChurnPredictions[Revenue_at_Risk])

Total Portfolio ACV = SUM(Customers[Annual_Contract_Value])

Pct Revenue at Risk = DIVIDE([Revenue at Risk (Actual)], [Total Portfolio ACV], 0)

Avg Customer Lifetime = AVERAGE(Customers[Years_With_Airtel])

Avg Satisfaction = AVERAGE(Customers[Customer_Satisfaction_Score])

High Risk Customers =
    CALCULATE(COUNTROWS(ChurnPredictions), ChurnPredictions[Risk_Category] IN {"High","Critical"})

Critical Risk Revenue =
    CALCULATE(SUM(ChurnPredictions[Revenue_at_Risk]), ChurnPredictions[Risk_Category] = "Critical")

Churn Rate by Reason =
    VAR ReasonTotal = CALCULATE(COUNTROWS(Customers), Customers[Churn] = 1)
    RETURN DIVIDE(COUNTROWS(Customers), ReasonTotal, 0)

MoM Churn Trend =
    CALCULATE(
        [Churned Customers],
        USERELATIONSHIP('DateTable'[Date], Customers[Churn_Date])
    )
```

## 3. Page-by-Page Design

### Page 1 — Executive Overview
**KPI cards** (top row): Total Customers · Active Customers · Churned Customers · Churn Rate · Revenue at Risk · Avg Customer Lifetime · Avg Satisfaction · High-Risk Customers (all measures above)

**Charts:**
- Donut: Churn vs Active (`[Total Customers]` split by `Customers[Churn]`)
- Line: Monthly Churn Trend (`[MoM Churn Trend]` by `DateTable[Month_Year]`)
- Clustered bar: Churn Rate by Customer Segment
- Clustered bar: Churn Rate by Industry (top 10)

### Page 2 — Churn Analysis
**Visuals:** Bar — Churn by Reason · Bar — Churn by Service (via `CustomerServices`/`ServiceCatalog`) · Bar — Churn by Contract Type · Bar — Churn by Company Size · Bar — Churn by Industry · Line — Churn Trend over time

**Slicers panel (left rail):** State · City · Industry · Customer Segment · Service (`ServiceCatalog[Service_Name]`) · Churn Reason · Contract Type

### Page 3 — Geographical Intelligence
- **Filled map** of India: `Customers[State]` on Location, `[Churn Rate]` on color saturation
- **Bubble map**: `Customers[City]` on Location, bubble size = `[Churned Customers]`, tooltip = `[Revenue at Risk (Actual)]`
- Table: Top 10 cities by churn (rate) and Top 10 cities by revenue lost, side by side
- Drill-down enabled: State → City (Power BI drill-through or hierarchy in the map visual)

### Page 4 — Service & Customer Experience
- Bar: Churn Rate by Airtel Service (from `CustomerServices`/`ServiceCatalog` join)
- Bar: Revenue Lost by Service
- Scatter: Network Satisfaction (x) vs Support Satisfaction (y), bubble size = Annual_Contract_Value, color = Churn
- Line/scatter: Downtime Hours vs Churn Rate (binned)
- Line/scatter: SLA Breaches vs Churn Rate
- Bar: Complaint frequency buckets vs Churn Rate

This page should visually answer: **"What is actually wrong with the service?"** — pair each quality metric directly against churn outcome.

### Page 5 — Churn Prediction & Retention
- Table (from `ChurnPredictions` joined to `Customers`): Company | Risk Category | Churn Probability | Revenue at Risk | Main Risk Factor | Recommended Action
  - **Conditional formatting**: Risk_Category background — Critical = dark red, High = orange, Medium = yellow, Low = green
  - **Conditional formatting**: Churn_Probability — data bars, red gradient
- KPI cards: High-Risk Customers · Critical-Risk Customers · Total Predicted Revenue at Risk
- Bar: Top churn drivers (from the model's feature importance — static image or table, since Power BI can't run the Python model live without a Python visual)
- Slicer: Risk_Category

## 4. Visual identity

- Primary accent: `#E60000` (red, evoking Airtel's brand family without copying their exact assets/logo)
- Background: white / light gray (`#F5F5F5`) cards on white canvas
- Font: Segoe UI (Power BI default) at consistent sizes — 24pt page titles, 12pt body
- Keep each page to 5-7 visuals max — this project intentionally limits chart count per page per the "don't overload dashboards" requirement

## 5. Publishing checklist
1. File → Options → Data Load → disable auto date/time (avoids duplicate hidden date tables)
2. Set `Churn` as a whole number, not a Boolean, before building visuals
3. Hide raw ID/technical columns from Report view (right-click → Hide in report view)
4. Add a bookmark-driven "Reset Filters" button on each page
5. Publish to Power BI Service → set up scheduled refresh if data source is live
