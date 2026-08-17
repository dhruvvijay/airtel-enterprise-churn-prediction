# Data Dictionary — Airtel Enterprise Customer Churn Dataset

**File:** `data/airtel_enterprise_churn.csv`
**Rows:** 25,000 synthetic enterprise customer records
**Columns:** 69
**Note:** This is a synthetically generated dataset inspired by publicly available Airtel Business service categories. It does not contain real Airtel customer data.

## Customer Information

| Column | Description | Data Type |
|---|---|---|
| Customer_ID | Unique customer identifier (e.g. AEC100001) | String |
| Company_Name | Synthetic company name | String |
| Company_Size | Small / Medium / Large / Enterprise | String |
| Industry | Industry vertical (17 categories) | String |
| Customer_Type | New Business / Existing Business / Upsell-Expansion | String |
| Customer_Segment | SMB / Mid-Market / Large Enterprise / Strategic-Key Account | String |
| Years_With_Airtel | Total customer relationship length, years | Float |

## Geography

| Column | Description | Data Type |
|---|---|---|
| State | Indian state (18 states covered) | String |
| City | Indian city (29 cities covered) | String |
| Region | North / South / East / West / Central | String |
| Pincode_Zone | Synthetic 6-digit zone code | Integer |

## Services Subscribed (binary flags, 1 = subscribed)

| Column | Category | Data Type |
|---|---|---|
| Internet_Leased_Line, Dedicated_Internet, MPLS_VPN, SD_WAN, Broadband, International_Private_Line | Connectivity | Integer (0/1) |
| Airtel_Cloud, Multi_Cloud_Connect, Cloud_Backup, Disaster_Recovery | Cloud | Integer (0/1) |
| Managed_Firewall, DDoS_Protection, Network_Security, Secure_Internet, Zero_Trust | Security | Integer (0/1) |
| Business_Voice, CPaaS, Enterprise_Messaging, Collaboration_Services | Communication | Integer (0/1) |
| IoT_Services | IoT | Integer (0/1) |
| Number_of_IoT_Devices | Count of connected IoT devices (0 if IoT_Services=0) | Integer |

## Service Usage

| Column | Description | Data Type |
|---|---|---|
| Number_of_Services | Total distinct services subscribed | Integer |
| Primary_Service | The customer's main/anchor service | String |
| Service_Tenure | Years on their primary service | Float |
| Monthly_Data_Usage_GB | Monthly data consumption | Float |
| Bandwidth_Mbps | Provisioned bandwidth | Float |
| Monthly_Bill | Monthly bill amount, INR | Float |
| Annual_Contract_Value | Annual contract value, INR (ACV) | Float |
| Contract_Type | Monthly / Annual / Multi-Year | String |
| Contract_Remaining_Months | Months left on current contract | Integer |

## Service Quality

| Column | Description | Data Type |
|---|---|---|
| Network_Uptime | % network availability | Float |
| Downtime_Hours | Total downtime hours (period) | Float |
| Number_of_Outages | Count of distinct outage events | Integer |
| Average_Latency_ms | Average network latency | Float |
| Packet_Loss_Percentage | % packet loss | Float |
| Service_Quality_Score | Composite quality score (20-100) | Float |
| Support_Response_Hours | Avg. time to first support response | Float |
| Support_Resolution_Hours | Avg. time to issue resolution | Float |
| Number_of_Complaints | Complaint count | Integer |
| Number_of_Service_Tickets | Total support tickets raised | Integer |
| SLA_Breaches | Count of SLA breach events (0-6) | Integer |
| Billing_Issues | Count of billing-related issues | Integer |
| Installation_Delay_Days | Days of delay during installation | Float |

## Customer Experience

| Column | Description | Data Type |
|---|---|---|
| Customer_Satisfaction_Score | Overall satisfaction, 1-10 | Float |
| NPS_Score | Net Promoter Score, -100 to 100 | Integer |
| Support_Satisfaction | Satisfaction with support, 1-10 | Float |
| Network_Satisfaction | Satisfaction with network, 1-10 | Float |
| Billing_Satisfaction | Satisfaction with billing, 1-10 | Float |
| Service_Satisfaction | Overall service satisfaction, 1-10 | Float |

## Competitor Signals

| Column | Description | Data Type |
|---|---|---|
| Competitor_Considered | Whether customer evaluated a competitor (0/1) | Integer |
| Competitor_Price_Difference | % price difference vs. competitor (negative = competitor cheaper) | Float |
| Competitor_Service_Rating | Customer's rating of competitor's service, 1-10 (0 if not considered) | Float |
| Competitor_Offer | Named competitor considered, or "None" | String |
| Competitor_Threat_Level | Low / Medium / High / Critical | String |

## Churn

| Column | Description | Data Type |
|---|---|---|
| Churn | 0 = Active, 1 = Churned | Integer |
| Churn_Date | Date of churn (blank if active) | Date (YYYY-MM-DD) |
| Churn_Reason | Dominant reason for churn (blank if active) | String |
| Churn_Category | Grouped reason category: Network & Reliability / Support & Service / Commercial / Competitive / Product Fit / Other (blank if active) | String |

## Generation methodology (for interview reference)

Churn is **not randomly assigned**. Each customer gets a latent churn score built as a weighted sum of standardized quality, experience, and commercial signals (downtime, outages, SLA breaches, support response time, complaints, billing issues, packet loss push churn up; satisfaction, NPS, uptime, tenure, service count, company size pull it down; competitor price pressure and near-end-of-contract timing push it up further), passed through a logistic function, with Gaussian noise added so the resulting classification problem is realistic rather than trivially separable. `Churn_Reason` is picked per-customer as whichever factor actually dominated that customer's score — it is derived from the same signals that drove the probability, not assigned independently.
