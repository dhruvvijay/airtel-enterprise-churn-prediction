# Interview Preparation — Airtel Enterprise Churn Project

## Technical Questions (20)

**1. Why did you generate synthetic data instead of using a public dataset like Telco Customer Churn?**
Public churn datasets (e.g. IBM Telco) are small (~7K rows), consumer-focused, and have already been analyzed thousands of times — they don't demonstrate the ability to design a realistic enterprise B2B dataset with logically consistent relationships. Building the generator myself let me control exactly how churn relates to service quality, and forced me to think about what actually drives enterprise telecom churn rather than just fitting a model to someone else's data.

**2. How did you make the churn label "not random"?**
I built a latent churn score as a weighted sum of z-scored features — downtime, outages, SLA breaches, support response time, complaints, billing issues push it up; satisfaction, NPS, uptime, tenure, service count, company size pull it down — passed through a logistic (sigmoid) function, then sampled a Bernoulli outcome from that probability. Gaussian noise is added to the logit so the resulting classification problem isn't trivially separable, which is what makes model comparison actually meaningful.

**3. Walk me through your data cleaning process.**
Since I control the generator, there were no missing-at-random values to impute — nulls only exist where they're structurally correct (Churn_Date/Reason/Category for active customers). I validated ranges (uptime capped 90-100%, satisfaction 1-10, no negative values), checked for duplicate Customer_IDs, and verified the churn rate and correlation signs matched the intended design before treating the dataset as final.

**4. What's the difference between Precision and Recall, and which mattered more here?**
Precision = of customers I predicted would churn, how many actually did. Recall = of customers who actually churned, how many did I catch. I prioritized Recall (and ROC-AUC) because the cost of *missing* a churning enterprise account (lost multi-year revenue) is usually much higher than the cost of an unnecessary retention call to a customer who wasn't going to leave. My selected model has 77.5% recall vs. 47.2% precision — a deliberate trade-off, not an oversight.

**5. Why is accuracy a misleading metric for this problem?**
With ~21% churn, a model that predicts "no churn" for everyone gets 78.6% accuracy while catching zero at-risk customers — completely useless for the business problem. I calculated that baseline explicitly in the notebook to make the point concrete rather than asserting it.

**6. Explain ROC-AUC in plain terms.**
It measures how well the model ranks customers by risk across every possible probability threshold — a value of 1.0 means a random churner always scores higher than a random non-churner; 0.5 means no better than a coin flip. My best model scores 0.848, meaning it's quite good at rank-ordering risk even before you pick a specific cutoff for "high risk."

**7. How did you handle the class imbalance?**
`class_weight="balanced"` in scikit-learn, which reweights the loss function inversely proportional to class frequency, rather than resampling (SMOTE, undersampling). I chose this over resampling because it doesn't create synthetic minority-class rows or throw away majority-class data — simpler and works well at this dataset size.

**8. Why Logistic Regression over the tree-based models, given trees are usually more powerful?**
On this dataset, Logistic Regression had the highest ROC-AUC (0.848) and by far the highest recall (0.775 vs. 0.418-0.586 for the tree models). The tree models had higher precision/accuracy but that's exactly the failure mode I was avoiding — they're more conservative and miss more actual churners. It's a reminder that "more powerful" model families don't automatically win on every metric or every dataset; you have to check what the business actually needs.

**9. What is multicollinearity, and did you encounter it here?**
Yes — directly. `Annual_Contract_Value` and `Monthly_Bill` were correlated at r=0.999 (since one is derived from the other), and a first pass at the model put both at the top of the "feature importance" ranking, which was misleading — it wasn't that revenue caused churn, it was that two nearly-identical columns split/inflated each other's coefficients. I checked a correlation matrix of near-duplicate feature pairs, dropped the redundant half of each (also did this for Bandwidth/Data Usage and Complaints/Tickets and Support Response/Resolution), and the real drivers (downtime, satisfaction, complaints) surfaced correctly afterward.

**10. How does Logistic Regression compute feature importance, versus Random Forest?**
Logistic Regression: after standardizing features (so they're on the same scale), the absolute value of each coefficient reflects how much that feature moves the log-odds of churn. Random Forest: importance is typically computed from how much each feature reduces impurity (e.g. Gini) across all the splits that use it, averaged across trees. I made sure my code used the *actual* method matching whichever model won, rather than reusing Random Forest's importances to explain a Logistic Regression model (a bug I caught and fixed during development).

**11. Why did you engineer a "Complaint_Satisfaction_Gap" feature?**
It captures inconsistency — a customer with many complaints but a comparatively small satisfaction score. Combining features can surface a signal neither raw feature captures alone: a customer who's satisfied on average but has spiking complaints looks different from one who's simply always dissatisfied. This is a modest additive feature in this dataset but demonstrates the reasoning behind feature engineering, not just running raw columns through a model.

**12. How would you explain a specific customer's churn prediction to a non-technical stakeholder (i.e., without SHAP)?**
SHAP wasn't installable in my build environment, so I built a lighter alternative: for Logistic Regression, a customer's per-feature contribution to the prediction is mathematically exact — it's just `coefficient x that customer's standardized value` for each feature, summed to the total logit. I surface the top 3 positive contributors (translated into plain language, e.g. "High network downtime") per customer. It's less general than SHAP (doesn't handle interaction effects as gracefully) but for a linear model it's the *exact* decomposition, not an approximation.

**13. What would you do differently with more time/compute?**
Install real XGBoost and SHAP for a more standard toolchain, run a full cross-validated grid search instead of the lightweight manual comparison I used (the sandboxed environment made a full GridSearchCV time out), and validate the risk tiers against a genuinely held-out future time period rather than a random train/test split.

**14. How did you validate that your risk tiers (Low/Medium/High/Critical) actually mean something?**
I checked that the *actual* churn rate observed within each predicted band increases monotonically from Low to Critical on the test set — that's the real test of a risk-tiering system, not just the aggregate ROC-AUC number.

**15. Explain a window function you used in SQL.**
`RANK() OVER (ORDER BY Churn_Rate_Pct DESC)` in the state-level churn query ranks every state by churn rate without collapsing the result set the way GROUP BY would — I can see every state's row *and* its rank in one query. I also used `ROW_NUMBER() OVER (PARTITION BY State ORDER BY Churn_Probability DESC)` to get the top-3 highest-risk active customers *within each state* for the retention team — PARTITION BY resets the ranking per group.

**16. Why normalize the service columns into a separate table instead of keeping 20 flag columns on the customer table?**
Wide binary-flag tables don't JOIN meaningfully and don't scale if Airtel launches new services (you'd need a schema migration for every new service). A `customer_services` bridge table plus a `service_catalog` dimension table lets me add new services as new rows, not new columns, and lets me demonstrate real JOIN-based analysis (service churn rate by category) rather than 20 nearly-identical WHERE clauses.

**17. What's a CTE and why use one over a subquery?**
A Common Table Expression (`WITH x AS (...)`) is a named, temporary result set scoped to one query — functionally similar to a subquery but more readable, especially when referenced multiple times or when chaining several logical steps (e.g. computing a percentile threshold, then filtering against it). I used one to compute the 75th percentile of complaint counts, then filtered customers against that threshold in the same query.

**18. How did you decide the train/test split strategy?**
80/20 stratified split on the target (`stratify=y`) to preserve the ~21% churn rate in both sets — critical for an imbalanced target, since a non-stratified split could easily produce a test set with a meaningfully different churn rate purely by chance, making metric comparisons noisy.

**19. What does `StandardScaler` do and why does only Logistic Regression need it?**
It rescales each feature to zero mean and unit variance. Logistic Regression's coefficients (and the convergence of its optimizer) are sensitive to feature scale — without scaling, a feature like Annual_Contract_Value (ranging into millions) would dominate a feature like Customer_Satisfaction_Score (1-10) purely due to scale, not real importance. Tree-based models (Decision Tree, Random Forest, Gradient Boosting) split on threshold comparisons within a single feature at a time, so they're scale-invariant and don't need it.

**20. In Power BI, what's the difference between a measure and a calculated column?**
A calculated column is computed once per row at data-refresh time and stored in the model (uses more memory, but can be used as a slicer/axis). A measure is computed on the fly at query time, in the context of whatever filters are currently applied (uses less memory, always reflects current filter context). My `[Churn Rate]` is a measure — it needs to recompute correctly whether you're looking at all customers or a filtered slice, which a static calculated column couldn't do as cleanly.

---

## Business Questions (15)

**1. Why do customers churn, according to your analysis?**
Overwhelmingly network and support quality, not price. Frequent Downtime alone accounts for 34% of all churn — more than the next two reasons combined — and the strongest statistical correlates of churn (downtime hours, outage count, uptime) are all service-quality metrics, not commercial ones.

**2. Which state has the worst churn?**
Depends on the question you're actually asking. Goa has the highest *rate* (25.1%) but is a small market (175 customers). Maharashtra has the highest *volume* (895 churned customers) but an average *rate* (21.0%). I'd tell a retention team both: Goa may signal a local service issue worth investigating; Maharashtra is where the largest absolute revenue-recovery opportunity is.

**3. Why might churn rate differ from churn volume, and why does that matter?**
Rate is churned/total within a group — it tells you how *risky* that segment is proportionally. Volume is the raw count — it tells you how much absolute impact fixing that segment would have. A small state can have a shocking rate off a tiny base; a huge state can have an unremarkable rate but dominate total churned customers. Prioritization should generally weight volume x revenue for resourcing decisions, and rate for diagnosing *where a systemic problem exists*.

**4. What is "revenue at risk" and how did you calculate it?**
Two versions: (1) *realized* — the total Annual Contract Value of customers who have already churned (₹1,405 Cr here), and (2) *forward-looking* — for currently active customers, `Annual_Contract_Value x model-predicted Churn_Probability`, summed across the base (₹2,759 Cr here). The second number is explicitly a model estimate, not a certainty — I'm careful to label it that way rather than present it as guaranteed loss.

**5. Which customers should Airtel contact first?**
Not simply "highest churn probability." I built a Retention Priority Score combining churn probability, revenue value, CLV, service breadth, and contract urgency — so a mid-probability, very-high-revenue account can rank above a near-certain-churn, low-value account. The idea is prioritizing *expected value protected*, not just risk.

**6. What service has the biggest problem?**
No single service is a dramatic outlier — churn rates across all 20 services cluster in a fairly tight 19.2%-21.0% band. Broadband has the highest churn rate (20.95%) and by far the largest revenue at risk (₹1,042 Cr), but mainly because it's the most widely subscribed service, not because it's uniquely broken. That itself is a finding: the problem is systemic (network/support quality), not isolated to one product.

**7. How would you reduce churn, practically?**
Revenue-weighted downtime reduction first (fix the biggest accounts' biggest problem), then support response-time improvement specifically (it correlates with churn more than resolution time), then a mechanical trigger for any account hitting 2+ SLA breaches to get proactive account management before they become a statistic.

**8. How would you validate your model before trusting it in production?**
Check that predicted risk tiers actually correspond to increasing real-world churn rates (I did this — it's monotonic Low→Critical on the test set), monitor for model drift as the real customer base changes over time, and — critically for this specific project — re-validate that XGBoost/SHAP-based versions (once available) don't materially change which features look most important, since the current pipeline used sklearn substitutes.

**9. What would you tell a customer success team to do differently based on this analysis?**
Stop treating contract length as a retention lever — multi-year contracts show almost no churn protection in this data. Instead, treat downtime and SLA breach counts as your early-warning system, and build a workflow that escalates automatically at 2+ breaches rather than waiting for the customer to complain.

**10. Which industries need more attention?**
Energy (24.1% churn) and Financial Services (23.4%) — both are also industries where enterprise customers typically have strict uptime/SLA requirements, which is consistent with the downtime-driven churn story rather than a coincidence.

**11. Is losing a Large Enterprise customer worse than losing 5 SMB customers?**
In pure revenue terms, likely yes — Large Enterprise accounts average far higher ACV, and the segment carries 58% of total portfolio value on just 20% of customers. But it's also the stickiest segment (17.9% churn vs. 23.7% for SMB), so losing one is also a stronger signal something has genuinely gone wrong, not just typical churn noise.

**12. How confident are you in the "revenue at risk" forward-looking number?**
It's a model estimate with real uncertainty — the model's ROC-AUC is 0.848, not 1.0, and Churn_Probability is a probability, not a certainty. I'd present ₹2,759 Cr as "the model's expected-value estimate under current conditions," not a guaranteed number, and I'd want to track how well it predicts actual churn over the next few reporting periods before leadership treats it as gospel.

**13. What's a "what-if" scenario you could run, and what would it tell you?**
E.g., "what happens to average churn probability if downtime is reduced 20% for everyone?" — `src/prediction.py` has a `what_if()` function that re-scores the population under such a scenario using the trained model. It's explicitly labeled a scenario estimate, not a guaranteed outcome, since the model was trained on observational data, not a controlled experiment — correlation-based estimates of intervention effects can overstate real-world impact.

**14. How would you measure whether a retention intervention actually worked?**
Ideally an A/B test: apply the recommended action to a random subset of flagged high-risk accounts and compare their actual churn rate against a held-out control group over the next few months — that's the only way to separate "the intervention worked" from "the model's risk estimate for this group was simply too high to begin with."

**15. What's the single most important takeaway from this project for a business leader?**
Enterprise churn here is predictable and it's not primarily about price — it's about downtime and how fast support responds when something breaks. The company doesn't need to compete on discounts to fix this; it needs to fix the underlying service reliability the model is pointing at, and the ₹1,326 Cr of revenue currently sitting in High/Critical risk accounts is the immediate, quantified reason to act.
