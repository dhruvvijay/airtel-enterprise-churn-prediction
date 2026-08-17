"""
generate_dataset.py
--------------------
Generates a synthetic Airtel Business-style enterprise customer dataset for
churn analysis. This is NOT real Airtel customer data. Service categories
are inspired by publicly known Airtel Business offerings (connectivity,
cloud, security, IoT, voice/collaboration). Company names, individual
customer records, and churn events are entirely fictional.

Design principle: churn is NOT random. It is driven by a weighted latent
score built from service-quality, support, pricing, competitor pressure,
and loyalty signals, passed through a logistic function, with noise added
so the resulting ML problem is realistic (not trivially separable).

Run:  python generate_dataset.py [n_rows] [output_path]
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "../data/airtel_enterprise_churn.csv"
SEED = 42
rng = np.random.default_rng(SEED)

TODAY = datetime(2026, 8, 17)

# ----------------------------------------------------------------------
# REFERENCE DATA
# ----------------------------------------------------------------------
INDUSTRIES = ["Banking", "Financial Services", "Insurance", "IT", "ITES",
              "Manufacturing", "Retail", "Healthcare", "Education", "Government",
              "Logistics", "Telecom", "E-commerce", "Media", "Energy",
              "Automotive", "Pharmaceuticals"]
# relative popularity of each industry among enterprise telecom customers
INDUSTRY_WEIGHTS = np.array([5, 4, 3, 12, 10, 9, 8, 6, 4, 5,
                              7, 2, 8, 3, 4, 5, 5], dtype=float)
INDUSTRY_WEIGHTS /= INDUSTRY_WEIGHTS.sum()

COMPANY_SIZES = ["Small", "Medium", "Large", "Enterprise"]
COMPANY_SIZE_WEIGHTS = [0.38, 0.32, 0.20, 0.10]

CUSTOMER_TYPES = ["New Business", "Existing Business", "Upsell/Expansion"]
CUSTOMER_TYPE_WEIGHTS = [0.20, 0.65, 0.15]

CUSTOMER_SEGMENTS = ["SMB", "Mid-Market", "Large Enterprise", "Strategic/Key Account"]

CITY_STATE_REGION = [
    ("Mumbai", "Maharashtra", "West"), ("Pune", "Maharashtra", "West"),
    ("Nagpur", "Maharashtra", "West"),
    ("Bengaluru", "Karnataka", "South"), ("Mysuru", "Karnataka", "South"),
    ("Chennai", "Tamil Nadu", "South"), ("Coimbatore", "Tamil Nadu", "South"),
    ("Delhi", "Delhi", "North"),
    ("Gurugram", "Haryana", "North"), ("Faridabad", "Haryana", "North"),
    ("Noida", "Uttar Pradesh", "North"), ("Lucknow", "Uttar Pradesh", "North"),
    ("Kanpur", "Uttar Pradesh", "North"),
    ("Hyderabad", "Telangana", "South"),
    ("Ahmedabad", "Gujarat", "West"), ("Surat", "Gujarat", "West"),
    ("Vadodara", "Gujarat", "West"),
    ("Kolkata", "West Bengal", "East"),
    ("Jaipur", "Rajasthan", "North"),
    ("Chandigarh", "Punjab", "North"),
    ("Kochi", "Kerala", "South"), ("Thiruvananthapuram", "Kerala", "South"),
    ("Indore", "Madhya Pradesh", "Central"), ("Bhopal", "Madhya Pradesh", "Central"),
    ("Patna", "Bihar", "East"),
    ("Bhubaneswar", "Odisha", "East"),
    ("Panaji", "Goa", "West"),
    ("Guwahati", "Assam", "East"),
    ("Visakhapatnam", "Andhra Pradesh", "South"),
]
CITY_WEIGHTS = np.array([14, 5, 2, 13, 2, 10, 3, 9, 6, 2, 5, 2, 2, 9,
                          5, 2, 2, 6, 3, 3, 4, 2, 3, 2, 2, 2, 1, 1, 2], dtype=float)
CITY_WEIGHTS /= CITY_WEIGHTS.sum()

CONTRACT_TYPES = ["Monthly", "Annual", "Multi-Year"]
CONTRACT_TYPE_WEIGHTS = [0.22, 0.48, 0.30]

CONNECTIVITY_SERVICES = ["Internet_Leased_Line", "Dedicated_Internet", "MPLS_VPN",
                          "SD_WAN", "Broadband", "International_Private_Line"]
CLOUD_SERVICES = ["Airtel_Cloud", "Multi_Cloud_Connect", "Cloud_Backup", "Disaster_Recovery"]
SECURITY_SERVICES = ["Managed_Firewall", "DDoS_Protection", "Network_Security",
                      "Secure_Internet", "Zero_Trust"]
COMMS_SERVICES = ["Business_Voice", "CPaaS", "Enterprise_Messaging", "Collaboration_Services"]
ALL_BINARY_SERVICES = CONNECTIVITY_SERVICES + CLOUD_SERVICES + SECURITY_SERVICES + COMMS_SERVICES

COMPETITORS = ["Jio", "Vi Business", "Tata Communications", "Vodafone Business", "None"]

CHURN_REASON_CATEGORY = {
    "Poor Network Quality": "Network & Reliability",
    "Frequent Downtime": "Network & Reliability",
    "Service Reliability": "Network & Reliability",
    "Slow Customer Support": "Support & Service",
    "Poor Service Quality": "Support & Service",
    "High Pricing": "Commercial",
    "Billing Problems": "Commercial",
    "Contract Issues": "Commercial",
    "Competitor Offer": "Competitive",
    "Better Technology Elsewhere": "Competitive",
    "Lack of Features": "Product Fit",
    "Migration to Cloud Provider": "Product Fit",
    "Internal IT Transformation": "Other",
    "Business Closure": "Other",
    "Other": "Other",
}

# Synthetic company name pools (clearly fictional)
NAME_PREFIX = ["Global", "Nova", "Prime", "Zenith", "Horizon", "Vertex", "Alpha",
               "NextGen", "Orion", "Apex", "Summit", "Bluewave", "Silverline",
               "Redstone", "Crestline", "Ironbridge", "Northgate", "Pioneer",
               "Meridian", "Cobalt", "Falcon", "Sterling", "Quantum", "Bright",
               "Anchor", "Lumen", "Skyline", "Vanguard", "Trueline", "Everest"]
NAME_SUFFIX = ["Solutions", "Retail India", "Manufacturing", "Financial Services",
               "Logistics", "Healthcare", "E-Commerce", "Technologies", "Systems",
               "Enterprises", "Industries", "Group", "Networks", "Labs", "Ventures",
               "Infra", "Digital", "Traders", "Pharma", "Energy"]

# ----------------------------------------------------------------------
# BASE ATTRIBUTES
# ----------------------------------------------------------------------
customer_id = np.array([f"AEC{100000 + i}" for i in range(N)])

company_name = np.array([
    f"{rng.choice(NAME_PREFIX)} {rng.choice(NAME_SUFFIX)}" for _ in range(N)
])
# de-duplicate lightly with a numeric suffix on collisions
seen = {}
company_name_final = []
for nm in company_name:
    if nm in seen:
        seen[nm] += 1
        company_name_final.append(f"{nm} {seen[nm]}")
    else:
        seen[nm] = 1
        company_name_final.append(nm)
company_name = np.array(company_name_final)

industry = rng.choice(INDUSTRIES, size=N, p=INDUSTRY_WEIGHTS)
company_size = rng.choice(COMPANY_SIZES, size=N, p=COMPANY_SIZE_WEIGHTS)
customer_type = rng.choice(CUSTOMER_TYPES, size=N, p=CUSTOMER_TYPE_WEIGHTS)

size_to_segment = {
    "Small": ["SMB", "SMB", "Mid-Market"],
    "Medium": ["SMB", "Mid-Market", "Mid-Market"],
    "Large": ["Mid-Market", "Large Enterprise", "Large Enterprise"],
    "Enterprise": ["Large Enterprise", "Large Enterprise", "Strategic/Key Account"],
}
customer_segment = np.array([rng.choice(size_to_segment[s]) for s in company_size])

city_idx = rng.choice(len(CITY_STATE_REGION), size=N, p=CITY_WEIGHTS)
city = np.array([CITY_STATE_REGION[i][0] for i in city_idx])
state = np.array([CITY_STATE_REGION[i][1] for i in city_idx])
region = np.array([CITY_STATE_REGION[i][2] for i in city_idx])
pincode_zone = rng.integers(100000, 855999, size=N)

years_with_airtel = np.round(rng.gamma(shape=2.1, scale=1.6, size=N), 1)
years_with_airtel = np.clip(years_with_airtel, 0.1, 15.0)

# ----------------------------------------------------------------------
# SERVICES
# ----------------------------------------------------------------------
size_rank = pd.Series(company_size).map({"Small": 0, "Medium": 1, "Large": 2, "Enterprise": 3}).values

# base probability of subscribing to a given service scales with company size
def service_flags(base_p, size_boost):
    p = np.clip(base_p + size_boost * size_rank / 3, 0.03, 0.97)
    return (rng.random(N) < p).astype(int)

service_matrix = {}
service_base_p = {
    "Internet_Leased_Line": 0.35, "Dedicated_Internet": 0.30, "MPLS_VPN": 0.28,
    "SD_WAN": 0.18, "Broadband": 0.45, "International_Private_Line": 0.08,
    "Airtel_Cloud": 0.22, "Multi_Cloud_Connect": 0.10, "Cloud_Backup": 0.20,
    "Disaster_Recovery": 0.12, "Managed_Firewall": 0.20, "DDoS_Protection": 0.14,
    "Network_Security": 0.25, "Secure_Internet": 0.18, "Zero_Trust": 0.07,
    "Business_Voice": 0.40, "CPaaS": 0.12, "Enterprise_Messaging": 0.15,
    "Collaboration_Services": 0.18,
}
for svc, base_p in service_base_p.items():
    service_matrix[svc] = service_flags(base_p, size_boost=0.35)

number_of_services = np.sum([service_matrix[s] for s in ALL_BINARY_SERVICES], axis=0)
# guarantee at least 1 service
zero_mask = number_of_services == 0
service_matrix["Broadband"] = np.where(zero_mask, 1, service_matrix["Broadband"])
number_of_services = np.sum([service_matrix[s] for s in ALL_BINARY_SERVICES], axis=0)

primary_service = np.array([
    rng.choice([s for s in ALL_BINARY_SERVICES if service_matrix[s][i] == 1])
    for i in range(N)
])

iot_services = service_flags(0.12, size_boost=0.30)
number_of_iot_devices = np.where(
    iot_services == 1,
    rng.integers(5, 2500, size=N),
    0
)

service_tenure = np.clip(years_with_airtel - rng.uniform(0, 0.6, size=N), 0.1, None)
service_tenure = np.round(service_tenure, 1)

bandwidth_base = {"Small": 50, "Medium": 200, "Large": 700, "Enterprise": 2000}
bandwidth_mbps = np.array([
    max(2, rng.gamma(shape=2.0, scale=bandwidth_base[s] / 2))
    for s in company_size
]).round(0)

monthly_data_usage_gb = np.round(bandwidth_mbps * rng.uniform(180, 420, size=N), 1)

contract_type = rng.choice(CONTRACT_TYPES, size=N, p=CONTRACT_TYPE_WEIGHTS)
contract_len_months = pd.Series(contract_type).map(
    {"Monthly": 1, "Annual": 12, "Multi-Year": 36}
).values
contract_remaining_months = np.array([
    rng.integers(0, cl + 1) if cl > 1 else rng.integers(0, 2)
    for cl in contract_len_months
])

# Monthly bill: base per company size + service count + bandwidth premium
size_bill_base = {"Small": 12000, "Medium": 45000, "Large": 160000, "Enterprise": 550000}
monthly_bill = np.array([size_bill_base[s] for s in company_size]).astype(float)
monthly_bill *= (1 + 0.14 * number_of_services)
monthly_bill *= (1 + bandwidth_mbps / bandwidth_mbps.max() * 0.6)
monthly_bill *= rng.uniform(0.85, 1.2, size=N)
monthly_bill = np.round(monthly_bill, -2)

annual_contract_value = np.round(monthly_bill * 12 * rng.uniform(0.95, 1.05, size=N), -2)

# ----------------------------------------------------------------------
# SERVICE QUALITY (these will drive churn)
# ----------------------------------------------------------------------
# Network uptime: mostly high (Beta skewed right), some bad performers
network_uptime = 100 - rng.beta(1.3, 18, size=N) * 100
network_uptime = np.clip(network_uptime, 90.0, 99.99).round(3)

downtime_hours = np.round((100 - network_uptime) * rng.uniform(3.0, 6.0, size=N), 1)
number_of_outages = np.round(downtime_hours / rng.uniform(1.2, 3.0, size=N)).astype(int)
number_of_outages = np.clip(number_of_outages, 0, None)

average_latency_ms = np.round(rng.gamma(shape=3.0, scale=8.0, size=N) + 5, 1)
packet_loss_pct = np.round(np.clip(rng.gamma(shape=1.4, scale=0.35, size=N), 0, 8), 2)

support_response_hours = np.round(np.clip(rng.gamma(shape=2.0, scale=3.2, size=N), 0.2, 72), 1)
support_resolution_hours = np.round(support_response_hours * rng.uniform(2.5, 6.0, size=N), 1)

number_of_complaints = rng.poisson(lam=np.clip(downtime_hours / 8 + packet_loss_pct / 2, 0.1, None))
number_of_service_tickets = number_of_complaints + rng.poisson(lam=1.2, size=N)

sla_breach_prob = np.clip(0.02 + downtime_hours / 300 + packet_loss_pct / 40, 0, 0.9)
sla_breaches = rng.binomial(n=6, p=sla_breach_prob)

billing_issues = rng.poisson(lam=0.6, size=N)
installation_delay_days = np.round(np.clip(rng.gamma(shape=1.6, scale=4.0, size=N), 0, 60), 0)

# composite service quality score (0-100), higher is better
service_quality_score = (
    network_uptime * 0.5
    + (100 - np.clip(average_latency_ms, 0, 100)) * 0.15
    + (100 - packet_loss_pct * 8) * 0.15
    + (100 - np.clip(support_response_hours, 0, 100)) * 0.20
)
service_quality_score = np.clip(service_quality_score, 20, 100).round(1)

# ----------------------------------------------------------------------
# CUSTOMER EXPERIENCE (correlated with quality metrics + noise)
# ----------------------------------------------------------------------
def score_from_quality(base_quality, noise_scale=8, lo=1, hi=10):
    raw = base_quality / 10 + rng.normal(0, noise_scale / 10, size=N)
    return np.clip(np.round(raw, 1), lo, hi)

network_satisfaction = score_from_quality(service_quality_score, noise_scale=9)
support_satisfaction = score_from_quality(100 - support_response_hours * 1.2, noise_scale=10)
billing_satisfaction = score_from_quality(100 - billing_issues * 12, noise_scale=10)
service_satisfaction = np.round(
    np.clip((network_satisfaction + support_satisfaction + billing_satisfaction) / 3
            + rng.normal(0, 0.4, size=N), 1, 10), 1)

customer_satisfaction_score = np.round(
    np.clip(service_satisfaction * 0.9 + rng.normal(0, 0.5, size=N), 1, 10), 1)

nps_score = np.round(np.clip(
    (customer_satisfaction_score - 5.5) * 22 + rng.normal(0, 12, size=N), -100, 100
)).astype(int)

# ----------------------------------------------------------------------
# COMPETITOR PRESSURE
# ----------------------------------------------------------------------
competitor_threat_base = np.clip(
    0.15 + (10 - customer_satisfaction_score) / 20 + rng.normal(0, 0.12, size=N), 0.02, 0.95
)
competitor_considered = (rng.random(N) < competitor_threat_base).astype(int)
competitor_offer = np.where(
    competitor_considered == 1,
    rng.choice(COMPETITORS[:-1], size=N),
    "None"
)
competitor_price_difference_pct = np.where(
    competitor_considered == 1,
    np.round(rng.uniform(-30, 5, size=N), 1),   # negative = competitor cheaper
    0.0
)
competitor_service_rating = np.where(
    competitor_considered == 1,
    np.round(rng.uniform(5.5, 9.0, size=N), 1),
    0.0
)
competitor_threat_level = pd.cut(
    competitor_threat_base, bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
    labels=["Low", "Medium", "High", "Critical"]
).astype(str)

# ----------------------------------------------------------------------
# CHURN LATENT SCORE (this is the heart of the "logical relationships" requirement)
# ----------------------------------------------------------------------
def z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)

churn_logit = (
    0.55 * z(downtime_hours)
    + 0.35 * z(number_of_outages)
    + 0.30 * z(sla_breaches)
    + 0.30 * z(support_response_hours)
    + 0.25 * z(number_of_complaints)
    + 0.20 * z(billing_issues)
    + 0.20 * z(packet_loss_pct)
    - 0.55 * z(customer_satisfaction_score)
    - 0.30 * z(nps_score)
    - 0.35 * z(network_uptime)
    - 0.20 * z(years_with_airtel)
    - 0.12 * z(number_of_services)
    - 0.18 * z(size_rank)  # larger enterprises get dedicated account mgmt -> stickier
    + 0.40 * (competitor_considered * (1 - competitor_price_difference_pct / 100))
    + 0.15 * z(np.where(contract_remaining_months <= 2, 1, 0))
    - 2.35  # intercept -> keeps overall churn rate realistic (~14-18%)
    + rng.normal(0, 1.1, size=N)  # noise so it's not perfectly separable
)
churn_probability_true = 1 / (1 + np.exp(-churn_logit))
churn = (rng.random(N) < churn_probability_true).astype(int)

# churn date: churned customers left at some point in the last ~24 months
days_ago = rng.integers(1, 730, size=N)
churn_date = np.where(
    churn == 1,
    [(TODAY - timedelta(days=int(d))).strftime("%Y-%m-%d") for d in days_ago],
    ""
)

# churn reason: pick the dominant contributing factor per churned customer
reason_pool = [
    "Poor Network Quality", "Frequent Downtime", "Slow Customer Support",
    "High Pricing", "Billing Problems", "Contract Issues", "Poor Service Quality",
    "Competitor Offer", "Better Technology Elsewhere", "Lack of Features",
    "Business Closure", "Service Reliability", "Migration to Cloud Provider",
    "Internal IT Transformation", "Other",
]

def pick_reason(i):
    factors = {
        "Poor Network Quality": z(network_uptime)[i] * -1 + z(packet_loss_pct)[i],
        "Frequent Downtime": z(downtime_hours)[i] + z(number_of_outages)[i],
        "Slow Customer Support": z(support_response_hours)[i] + z(support_resolution_hours)[i],
        "High Pricing": (competitor_price_difference_pct[i] < -10) * 2.0 + rng.normal(0, 0.3),
        "Billing Problems": z(billing_issues)[i],
        "Contract Issues": (contract_remaining_months[i] <= 2) * 1.8 + rng.normal(0, 0.3),
        "Poor Service Quality": (100 - service_quality_score[i]) / 20,
        "Competitor Offer": competitor_considered[i] * 1.6 + rng.normal(0, 0.3),
        "Better Technology Elsewhere": competitor_service_rating[i] / 3 if competitor_considered[i] else 0,
        "Lack of Features": rng.normal(0.3, 0.4),
        "Business Closure": rng.normal(-1.5, 0.5),
        "Service Reliability": z(sla_breaches)[i],
        "Migration to Cloud Provider": rng.normal(0.1, 0.4),
        "Internal IT Transformation": rng.normal(0.0, 0.4),
        "Other": rng.normal(-1.0, 0.4),
    }
    return max(factors, key=factors.get)

churn_reason = np.array([pick_reason(i) if churn[i] == 1 else "" for i in range(N)])
churn_category = np.array([
    CHURN_REASON_CATEGORY.get(r, "") if r else "" for r in churn_reason
])

# ----------------------------------------------------------------------
# ASSEMBLE DATAFRAME
# ----------------------------------------------------------------------
df = pd.DataFrame({
    "Customer_ID": customer_id,
    "Company_Name": company_name,
    "Company_Size": company_size,
    "Industry": industry,
    "Customer_Type": customer_type,
    "Customer_Segment": customer_segment,
    "Years_With_Airtel": years_with_airtel,
    "State": state,
    "City": city,
    "Region": region,
    "Pincode_Zone": pincode_zone,
    **{svc: service_matrix[svc] for svc in ALL_BINARY_SERVICES},
    "IoT_Services": iot_services,
    "Number_of_IoT_Devices": number_of_iot_devices,
    "Number_of_Services": number_of_services,
    "Primary_Service": primary_service,
    "Service_Tenure": service_tenure,
    "Monthly_Data_Usage_GB": monthly_data_usage_gb,
    "Bandwidth_Mbps": bandwidth_mbps,
    "Monthly_Bill": monthly_bill,
    "Annual_Contract_Value": annual_contract_value,
    "Contract_Type": contract_type,
    "Contract_Remaining_Months": contract_remaining_months,
    "Network_Uptime": network_uptime,
    "Downtime_Hours": downtime_hours,
    "Number_of_Outages": number_of_outages,
    "Average_Latency_ms": average_latency_ms,
    "Packet_Loss_Percentage": packet_loss_pct,
    "Service_Quality_Score": service_quality_score,
    "Support_Response_Hours": support_response_hours,
    "Support_Resolution_Hours": support_resolution_hours,
    "Number_of_Complaints": number_of_complaints,
    "Number_of_Service_Tickets": number_of_service_tickets,
    "SLA_Breaches": sla_breaches,
    "Billing_Issues": billing_issues,
    "Installation_Delay_Days": installation_delay_days,
    "Customer_Satisfaction_Score": customer_satisfaction_score,
    "NPS_Score": nps_score,
    "Support_Satisfaction": support_satisfaction,
    "Network_Satisfaction": network_satisfaction,
    "Billing_Satisfaction": billing_satisfaction,
    "Service_Satisfaction": service_satisfaction,
    "Competitor_Considered": competitor_considered,
    "Competitor_Price_Difference": competitor_price_difference_pct,
    "Competitor_Service_Rating": competitor_service_rating,
    "Competitor_Offer": competitor_offer,
    "Competitor_Threat_Level": competitor_threat_level,
    "Churn": churn,
    "Churn_Date": churn_date,
    "Churn_Reason": churn_reason,
    "Churn_Category": churn_category,
})

df.to_csv(OUT_PATH, index=False)

print(f"Generated {len(df):,} rows -> {OUT_PATH}")
print(f"Overall churn rate: {df['Churn'].mean()*100:.2f}%")
print("\nChurn rate by company size:")
print(df.groupby("Company_Size")["Churn"].mean().round(3) * 100)
print("\nCorrelation of key drivers with churn:")
drivers = ["Downtime_Hours", "SLA_Breaches", "Support_Response_Hours",
           "Number_of_Complaints", "Customer_Satisfaction_Score", "NPS_Score",
           "Network_Uptime", "Years_With_Airtel"]
print(df[drivers + ["Churn"]].corr()["Churn"].sort_values(ascending=False))
