-- ============================================================================
-- airtel_churn_analysis.sql
-- Airtel Enterprise Customer Churn — Business Analysis Queries (MySQL 8.0+)
-- Run after 00_schema_and_load.sql has loaded the data.
-- ============================================================================
USE airtel_churn;

-- ============================================================================
-- 1. TOTAL CUSTOMERS
-- ============================================================================
SELECT COUNT(*) AS Total_Customers
FROM customers;

-- ============================================================================
-- 2. TOTAL CHURN
-- ============================================================================
SELECT SUM(Churn) AS Total_Churned_Customers
FROM customers;

-- ============================================================================
-- 3. CHURN RATE
-- ============================================================================
SELECT
    COUNT(*)                                   AS Total_Customers,
    SUM(Churn)                                 AS Churned_Customers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)    AS Churn_Rate_Pct
FROM customers;

-- ============================================================================
-- 4. CHURN BY STATE  (rate + volume + window-ranked)
-- ============================================================================
WITH state_stats AS (
    SELECT
        State,
        COUNT(*)                                AS Total_Customers,
        SUM(Churn)                               AS Churned_Customers,
        ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)  AS Churn_Rate_Pct,
        SUM(CASE WHEN Churn = 1 THEN Annual_Contract_Value ELSE 0 END) AS Revenue_at_Risk
    FROM customers
    GROUP BY State
)
SELECT
    State, Total_Customers, Churned_Customers, Churn_Rate_Pct, Revenue_at_Risk,
    RANK() OVER (ORDER BY Churn_Rate_Pct DESC)      AS Rank_by_Rate,
    RANK() OVER (ORDER BY Churned_Customers DESC)   AS Rank_by_Volume
FROM state_stats
ORDER BY Churn_Rate_Pct DESC;

-- ============================================================================
-- 5. CHURN BY CITY (top 15 by rate, min 50 customers to avoid small-sample noise)
-- ============================================================================
WITH city_stats AS (
    SELECT
        City, State,
        COUNT(*)                                AS Total_Customers,
        SUM(Churn)                               AS Churned_Customers,
        ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)  AS Churn_Rate_Pct
    FROM customers
    GROUP BY City, State
    HAVING COUNT(*) >= 50
)
SELECT *
FROM city_stats
ORDER BY Churn_Rate_Pct DESC
LIMIT 15;

-- ============================================================================
-- 6. CHURN BY SERVICE (JOIN across customers -> customer_services -> service_catalog)
-- ============================================================================
SELECT
    sc.Service_Category,
    cs.Service_Name,
    COUNT(*)                                                            AS Subscribers,
    SUM(c.Churn)                                                        AS Churned_Subscribers,
    ROUND(SUM(c.Churn) * 100.0 / COUNT(*), 2)                           AS Churn_Rate_Pct,
    SUM(CASE WHEN c.Churn = 1 THEN c.Annual_Contract_Value ELSE 0 END)  AS Revenue_at_Risk
FROM customer_services cs
JOIN customers c        ON c.Customer_ID = cs.Customer_ID
JOIN service_catalog sc ON sc.Service_Name = cs.Service_Name
GROUP BY sc.Service_Category, cs.Service_Name
ORDER BY Churn_Rate_Pct DESC;

-- ============================================================================
-- 7. CHURN BY INDUSTRY
-- ============================================================================
SELECT
    Industry,
    COUNT(*)                                  AS Total_Customers,
    SUM(Churn)                                 AS Churned_Customers,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)    AS Churn_Rate_Pct,
    SUM(CASE WHEN Churn = 1 THEN Annual_Contract_Value ELSE 0 END) AS Revenue_at_Risk
FROM customers
GROUP BY Industry
ORDER BY Churn_Rate_Pct DESC;

-- ============================================================================
-- 8. CHURN BY REASON (with grouped category via CASE, plus % share)
-- ============================================================================
SELECT
    Churn_Reason,
    CASE
        WHEN Churn_Reason IN ('Poor Network Quality','Frequent Downtime','Service Reliability')
            THEN 'Network & Reliability'
        WHEN Churn_Reason IN ('Slow Customer Support','Poor Service Quality')
            THEN 'Support & Service'
        WHEN Churn_Reason IN ('High Pricing','Billing Problems','Contract Issues')
            THEN 'Commercial'
        WHEN Churn_Reason IN ('Competitor Offer','Better Technology Elsewhere')
            THEN 'Competitive'
        WHEN Churn_Reason IN ('Lack of Features','Migration to Cloud Provider')
            THEN 'Product Fit'
        ELSE 'Other'
    END AS Churn_Category,
    COUNT(*) AS Customers,
    ROUND(COUNT(*) * 100.0 / (SELECT SUM(Churn) FROM customers), 2) AS Pct_of_Churn
FROM customers
WHERE Churn = 1
GROUP BY Churn_Reason, Churn_Category
ORDER BY Customers DESC;

-- ============================================================================
-- 9. REVENUE LOST (total + by segment)
-- ============================================================================
SELECT
    ROUND(SUM(CASE WHEN Churn = 1 THEN Annual_Contract_Value ELSE 0 END), 2) AS Total_Revenue_Lost,
    ROUND(SUM(Annual_Contract_Value), 2)                                     AS Total_Portfolio_ACV,
    ROUND(SUM(CASE WHEN Churn = 1 THEN Annual_Contract_Value ELSE 0 END)
          * 100.0 / SUM(Annual_Contract_Value), 2)                          AS Pct_Revenue_Lost
FROM customers;

-- ============================================================================
-- 10. HIGHEST-VALUE CHURNED CUSTOMERS (top 15 by ACV)
-- ============================================================================
SELECT
    Customer_ID, Company_Name, Industry, State, City,
    Number_of_Services, Annual_Contract_Value, Churn_Reason, Churn_Date
FROM customers
WHERE Churn = 1
ORDER BY Annual_Contract_Value DESC
LIMIT 15;

-- ============================================================================
-- 11. ACTIVE CUSTOMERS (count + profile summary)
-- ============================================================================
SELECT
    COUNT(*)                                          AS Active_Customers,
    ROUND(AVG(Years_With_Airtel), 2)                  AS Avg_Tenure_Years,
    ROUND(AVG(Customer_Satisfaction_Score), 2)        AS Avg_Satisfaction,
    ROUND(AVG(Network_Uptime), 2)                     AS Avg_Network_Uptime,
    ROUND(AVG(Number_of_Services), 2)                 AS Avg_Services_Subscribed,
    ROUND(SUM(Annual_Contract_Value), 2)              AS Total_Active_ACV
FROM customers
WHERE Churn = 0;

-- ============================================================================
-- 12. HIGH-RISK CUSTOMERS (from ML pipeline output — requires churn_predictions
--     table to be loaded; see 00_schema_and_load.sql)
-- ============================================================================
SELECT
    c.Customer_ID, c.Company_Name, c.Industry, c.State, c.City,
    cp.Churn_Probability, cp.Risk_Category, cp.Revenue_at_Risk,
    cp.Main_Risk_Factor, cp.Recommended_Action
FROM churn_predictions cp
JOIN customers c ON c.Customer_ID = cp.Customer_ID
WHERE cp.Risk_Category IN ('High', 'Critical')
ORDER BY cp.Revenue_at_Risk DESC
LIMIT 50;

-- ============================================================================
-- 13. CUSTOMERS WITH MULTIPLE COMPLAINTS (above the 75th percentile, via CTE)
-- ============================================================================
WITH complaint_pctile AS (
    SELECT Number_of_Complaints,
           PERCENT_RANK() OVER (ORDER BY Number_of_Complaints) AS pct_rank
    FROM customers
)
SELECT
    c.Customer_ID, c.Company_Name, c.Number_of_Complaints, c.Number_of_Service_Tickets,
    c.Churn, c.Customer_Satisfaction_Score
FROM customers c
WHERE c.Number_of_Complaints >= (
    SELECT MIN(Number_of_Complaints) FROM complaint_pctile WHERE pct_rank >= 0.75
)
ORDER BY c.Number_of_Complaints DESC
LIMIT 100;

-- ============================================================================
-- 14. CUSTOMERS WITH SLA BREACHES (grouped, with churn outcome)
-- ============================================================================
SELECT
    SLA_Breaches,
    COUNT(*)                                 AS Customers,
    SUM(Churn)                                AS Churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)   AS Churn_Rate_Pct
FROM customers
GROUP BY SLA_Breaches
HAVING SLA_Breaches > 0
ORDER BY SLA_Breaches DESC;

-- ============================================================================
-- 15. AVERAGE SATISFACTION: CHURNED vs ACTIVE (side-by-side comparison)
-- ============================================================================
SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Active' END AS Customer_Status,
    COUNT(*)                                        AS Customers,
    ROUND(AVG(Customer_Satisfaction_Score), 2)      AS Avg_Satisfaction,
    ROUND(AVG(NPS_Score), 2)                        AS Avg_NPS,
    ROUND(AVG(Network_Satisfaction), 2)             AS Avg_Network_Satisfaction,
    ROUND(AVG(Support_Satisfaction), 2)             AS Avg_Support_Satisfaction,
    ROUND(AVG(Downtime_Hours), 2)                   AS Avg_Downtime_Hours,
    ROUND(AVG(Support_Response_Hours), 2)           AS Avg_Support_Response_Hours
FROM customers
GROUP BY Churn;

-- ============================================================================
-- BONUS: Retention Priority — high revenue + high churn probability, ranked
-- within each state using a window function (useful for the retention team)
-- ============================================================================
WITH ranked AS (
    SELECT
        c.Customer_ID, c.Company_Name, c.State, c.Annual_Contract_Value,
        cp.Churn_Probability, cp.Risk_Category,
        ROW_NUMBER() OVER (PARTITION BY c.State ORDER BY cp.Churn_Probability DESC) AS state_risk_rank
    FROM customers c
    JOIN churn_predictions cp ON cp.Customer_ID = c.Customer_ID
    WHERE c.Churn = 0  -- still-active customers worth prioritizing for retention
)
SELECT *
FROM ranked
WHERE state_risk_rank <= 3
ORDER BY State, state_risk_rank;
