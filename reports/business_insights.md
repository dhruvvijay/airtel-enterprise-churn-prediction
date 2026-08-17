# Business Insights — Airtel Enterprise Customer Churn

*All figures below are computed directly from `data/airtel_enterprise_churn.csv` (25,000 synthetic enterprise customers) — see `src/compute_insights.py`. Currency figures are in INR; "Cr" = crore (10 million).*

## Executive Summary

Of 25,000 enterprise customers, **5,358 have churned (21.43%)**, putting **₹1,405 Cr of annual contract value at risk (17.5% of the ₹8,009 Cr total portfolio)**. Churn here is overwhelmingly a **network and support quality problem**, not primarily a pricing problem: the single largest reason customers leave is frequent downtime (34% of all churn), and the strongest statistical drivers of churn are downtime hours, outage count, and network uptime — not price or contract type.

## 1. Biggest Churn Driver

**Frequent Downtime** accounts for 1,820 of 5,358 churned customers (33.97%) — more than the next two reasons combined. **Slow Customer Support** is second (969 customers, 18.09%). Together, network-and-support-quality issues (Frequent Downtime + Poor Network Quality + Slow Customer Support + Poor Service Quality + Service Reliability) account for the clear majority of all churn — this is a service delivery problem the network and support organizations can directly act on, not primarily a commercial/pricing problem.

Correlation analysis backs this up: `Downtime_Hours` (+0.40), `Number_of_Outages` (+0.38), and `Network_Uptime` (-0.38) are the three strongest correlates of churn in the entire dataset — stronger than satisfaction scores, NPS, tenure, or number of services subscribed.

## 2. Biggest Revenue Risk

**₹1,405 Cr in annual contract value has already been lost to churned customers.** The highest-value single churned accounts are concentrated in Retail, Manufacturing, and Education, with the three largest churned contracts each worth ₹270-290 lakh/year — and two of the three cite "Better Technology Elsewhere" as the reason, meaning the account didn't just leave over price, they believed a competitor's underlying technology was better.

## 3. Worst-Performing Service

**Broadband** has both the highest churn rate among major services (20.95%) and by far the largest absolute revenue at risk (₹1,042 Cr), simply because it's the most widely subscribed service (14,353 customers). **Cloud_Backup** (20.89%) and **Network_Security** (20.81%) follow closely. Interestingly, churn rates across services cluster fairly tightly (19.2%-21.0%) — no single service is a dramatic outlier, which suggests the underlying problem (network reliability, support responsiveness) cuts across the whole service portfolio rather than being isolated to one product line.

## 4. Worst-Performing City / State

**By rate:** Goa (25.14%) and Andhra Pradesh (23.67%) have the highest churn rates, though both are relatively small markets (175 and 376 customers respectively) — a handful of accounts moves the rate a lot here, so this needs to be read alongside volume.

**By volume:** Maharashtra (895 churned customers) and Karnataka (653) carry the largest absolute churn simply because they're Airtel's largest enterprise markets — their churn *rates* (21.0% and 21.6%) are actually close to the portfolio average. **This is the single most important distinction for a retention team**: a state with a high rate but a small base (Goa) signals a possibly acute local service issue worth investigating; a state with average rate but huge volume (Maharashtra) is where the largest absolute revenue recovery opportunity sits.

## 5. Highest-Risk Industry

**Energy** has the highest industry churn rate at 24.11% (238 of 987 customers), followed by **Financial Services** (23.35%) and **Logistics** (22.48%). Both Energy and Financial Services are industries with typically strict uptime/SLA requirements, consistent with the downtime-driven churn story.

## 6. Most Valuable Churned Customer Segment vs. Most Loyal Segment

**Large Enterprise** customers make up only 20% of the customer base (5,009 of 25,000) but **58% of total portfolio value** (₹4,651 Cr of ₹8,009 Cr) — and they churn at a below-average 17.89% rate. **Strategic/Key Accounts churn least of all** (16.77%). At the other end, **SMB customers churn most** (23.73%) but represent only ~5% of portfolio value. The practical read: retention effort concentrated on Large Enterprise and Strategic accounts protects far more revenue per customer saved, even though they're already the stickiest segment — losing even a few of them matters more than losing many SMB accounts.

## 7. What Separates Loyal Customers from Churned Ones

| Metric | Active (Loyal) | Churned | Gap |
|---|---:|---:|---:|
| Avg downtime hours | 21.7h | 37.6h | Churned customers experience **73% more downtime** |
| Avg support response time | 6.1h | 7.6h | 25% slower |
| Avg network satisfaction | 9.08/10 | 8.79/10 | Modest but consistent gap |
| Avg service quality score | 91.4 | 89.4 | Consistent gap |
| Avg number of services | 6.32 | 5.90 | Loyal customers use more services |
| Avg annual contract value | ₹33.6 lakh | ₹26.2 lakh | Loyal customers are higher-value on average |
| % on multi-year contracts | 30.7% | 29.4% | **Barely different** — contract length is a weak predictor here |

The clearest, most actionable gap is downtime — not contract type, not even satisfaction score by itself. **Customers don't leave because they signed a short contract; they leave because the network let them down.**

## 8. What Airtel Could Do to Reduce Churn

1. **Attack downtime for high-revenue accounts first.** Revenue-weighted downtime reduction (prioritizing NOC/field response for large accounts with elevated downtime) targets the single largest churn driver where it matters most financially.
2. **Fix support responsiveness, not just resolution.** Response time correlates with churn more than resolution time — customers seem to judge Airtel on how fast someone *acknowledges* the problem, not just how fast it's fixed.
3. **Watch SLA-breach repeaters.** Customers with 2+ SLA breaches show a clear step-up in churn — this is a cheap, mechanical trigger for proactive account management.
4. **Don't rely on contract length as a retention lever.** Multi-year contracts show almost no protective effect here — retention needs to come from service quality, not lock-in.
5. **Watch Energy and Financial Services accounts more closely** — both have above-average churn and are relatively high-stakes/high-visibility industries to lose.

## 9. Revenue at Risk — Forward-Looking (from the ML model)

Beyond the ₹1,405 Cr already lost, the trained churn model flags **6,977 currently-active customers as Critical or High risk** (3,094 Critical, 3,883 High), representing a further **₹1,326 Cr in probability-weighted revenue at risk** among just that High+Critical group (`Annual_Contract_Value x Churn_Probability`, summed). Across the full active base the model estimates **₹2,759 Cr** in total probability-weighted revenue at risk. See `models/churn_predictions.csv` and the Streamlit dashboard's Risk & Retention tab for the prioritized list.
