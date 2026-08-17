-- ============================================================================
-- 00_schema_and_load.sql
-- Airtel Enterprise Customer Churn — Database Schema & Data Load
-- ============================================================================
-- Creates a normalized-enough schema (one wide fact table + a service
-- dimension + a service-subscription bridge table) so the analysis script
-- can demonstrate real JOINs, not just single-table aggregation.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS airtel_churn;
USE airtel_churn;

-- ----------------------------------------------------------------------------
-- Main customer fact table (wide — mirrors airtel_enterprise_churn.csv)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    Customer_ID                 VARCHAR(20) PRIMARY KEY,
    Company_Name                VARCHAR(150),
    Company_Size                VARCHAR(20),
    Industry                    VARCHAR(50),
    Customer_Type                VARCHAR(30),
    Customer_Segment            VARCHAR(40),
    Years_With_Airtel           DECIMAL(5,1),
    State                       VARCHAR(50),
    City                        VARCHAR(50),
    Region                      VARCHAR(20),
    Pincode_Zone                INT,
    Number_of_IoT_Devices       INT,
    Number_of_Services          INT,
    Primary_Service              VARCHAR(50),
    Service_Tenure               DECIMAL(5,1),
    Monthly_Data_Usage_GB       DECIMAL(12,1),
    Bandwidth_Mbps              DECIMAL(10,1),
    Monthly_Bill                DECIMAL(14,2),
    Annual_Contract_Value       DECIMAL(14,2),
    Contract_Type                VARCHAR(20),
    Contract_Remaining_Months   INT,
    Network_Uptime              DECIMAL(6,3),
    Downtime_Hours              DECIMAL(8,1),
    Number_of_Outages           INT,
    Average_Latency_ms          DECIMAL(8,1),
    Packet_Loss_Percentage      DECIMAL(5,2),
    Service_Quality_Score       DECIMAL(6,1),
    Support_Response_Hours      DECIMAL(6,1),
    Support_Resolution_Hours    DECIMAL(6,1),
    Number_of_Complaints        INT,
    Number_of_Service_Tickets   INT,
    SLA_Breaches                 INT,
    Billing_Issues                INT,
    Installation_Delay_Days     DECIMAL(6,1),
    Customer_Satisfaction_Score DECIMAL(4,1),
    NPS_Score                   INT,
    Support_Satisfaction        DECIMAL(4,1),
    Network_Satisfaction        DECIMAL(4,1),
    Billing_Satisfaction        DECIMAL(4,1),
    Service_Satisfaction        DECIMAL(4,1),
    Competitor_Considered       TINYINT,
    Competitor_Price_Difference DECIMAL(6,1),
    Competitor_Service_Rating   DECIMAL(4,1),
    Competitor_Offer            VARCHAR(50),
    Competitor_Threat_Level     VARCHAR(20),
    Churn                       TINYINT,
    Churn_Date                  DATE NULL,
    Churn_Reason                 VARCHAR(60),
    Churn_Category               VARCHAR(40),
    INDEX idx_state (State),
    INDEX idx_city (City),
    INDEX idx_industry (Industry),
    INDEX idx_churn (Churn)
);

-- ----------------------------------------------------------------------------
-- Service dimension table
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS service_catalog;
CREATE TABLE service_catalog (
    Service_Name     VARCHAR(50) PRIMARY KEY,
    Service_Category VARCHAR(30)
);

-- ----------------------------------------------------------------------------
-- Customer <-> Service bridge table (long format — one row per subscription)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS customer_services;
CREATE TABLE customer_services (
    Customer_ID  VARCHAR(20),
    Service_Name VARCHAR(50),
    PRIMARY KEY (Customer_ID, Service_Name),
    FOREIGN KEY (Customer_ID) REFERENCES customers(Customer_ID),
    FOREIGN KEY (Service_Name) REFERENCES service_catalog(Service_Name)
);

-- ----------------------------------------------------------------------------
-- ML output table — populated after running the churn prediction pipeline
-- (notebooks/02_Churn_Prediction_Model.ipynb exports models/churn_predictions.csv)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS churn_predictions;
CREATE TABLE churn_predictions (
    Customer_ID        VARCHAR(20) PRIMARY KEY,
    Churn_Probability  DECIMAL(6,4),
    Risk_Category       VARCHAR(20),
    Revenue_at_Risk     DECIMAL(14,2),
    Main_Risk_Factor    VARCHAR(60),
    Recommended_Action  VARCHAR(120),
    FOREIGN KEY (Customer_ID) REFERENCES customers(Customer_ID)
);

-- ----------------------------------------------------------------------------
-- Load data
-- Adjust file paths to your local MySQL secure_file_priv directory, or use
-- MySQL Workbench's "Table Data Import Wizard" instead of LOAD DATA INFILE.
-- ----------------------------------------------------------------------------
LOAD DATA LOCAL INFILE '../data/airtel_enterprise_churn.csv'
INTO TABLE customers
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Customer_ID, Company_Name, Company_Size, Industry, Customer_Type, Customer_Segment,
 Years_With_Airtel, State, City, Region, Pincode_Zone,
 @Internet_Leased_Line, @Dedicated_Internet, @MPLS_VPN, @SD_WAN, @Broadband,
 @International_Private_Line, @Airtel_Cloud, @Multi_Cloud_Connect, @Cloud_Backup,
 @Disaster_Recovery, @Managed_Firewall, @DDoS_Protection, @Network_Security,
 @Secure_Internet, @Zero_Trust, @Business_Voice, @CPaaS, @Enterprise_Messaging,
 @Collaboration_Services, @IoT_Services, Number_of_IoT_Devices, Number_of_Services,
 Primary_Service, Service_Tenure, Monthly_Data_Usage_GB, Bandwidth_Mbps, Monthly_Bill,
 Annual_Contract_Value, Contract_Type, Contract_Remaining_Months, Network_Uptime,
 Downtime_Hours, Number_of_Outages, Average_Latency_ms, Packet_Loss_Percentage,
 Service_Quality_Score, Support_Response_Hours, Support_Resolution_Hours,
 Number_of_Complaints, Number_of_Service_Tickets, SLA_Breaches, Billing_Issues,
 Installation_Delay_Days, Customer_Satisfaction_Score, NPS_Score, Support_Satisfaction,
 Network_Satisfaction, Billing_Satisfaction, Service_Satisfaction, Competitor_Considered,
 Competitor_Price_Difference, Competitor_Service_Rating, Competitor_Offer,
 Competitor_Threat_Level, Churn, @Churn_Date, Churn_Reason, Churn_Category)
SET Churn_Date = NULLIF(@Churn_Date, '');
-- Note: the 20 binary service flag columns (@Internet_Leased_Line etc.) are captured
-- into user variables and intentionally NOT loaded into `customers` — they are
-- normalized into `customer_services` below instead, so the schema demonstrates
-- a proper fact/dimension/bridge structure rather than 20 flag columns in one table.

LOAD DATA LOCAL INFILE '../sql/service_catalog.csv'
INTO TABLE service_catalog
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '../sql/customer_services.csv'
INTO TABLE customer_services
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- After running the ML notebook (which exports models/churn_predictions.csv):
-- LOAD DATA LOCAL INFILE '../models/churn_predictions.csv'
-- INTO TABLE churn_predictions
-- FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
-- LINES TERMINATED BY '\n'
-- IGNORE 1 ROWS;

SELECT COUNT(*) AS customers_loaded FROM customers;
SELECT COUNT(*) AS service_subscriptions_loaded FROM customer_services;
