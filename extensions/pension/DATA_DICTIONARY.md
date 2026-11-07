# Data dictionary and synthetic provenance

All records are fabricated by `src/generate_data.py`, seed 2026. Input data are not sourced from a pension administrator and contain no real identities. No jurisdiction-specific pension rules are encoded.

| Field | Meaning |
|---|---|
| `member_id` | Unique fabricated member ID; not a feature |
| `snapshot_date` | Month-end scoring cutoff; not a predictor |
| `age` | Simulated age 25–60; audit only |
| `tenure_months` | Membership duration as of snapshot |
| `account_balance` | Snapshot balance in illustrative USD |
| `monthly_contribution` | Current monthly contribution amount, USD |
| `months_since_contribution` | Months since most recent contribution as of snapshot |
| `employment_status` | Employed / Self-employed / Not contributing; simulation categories |
| `portal_visits_90d` | Portal visits in preceding 90 days |
| `complaints_90d` | Service complaints in preceding 90 days |
| `transfer_inquiry_90d` | Pre-snapshot inquiry about transfer, 0/1 |
| `satisfaction_score` | Survey score 1–5; 6% missing, imputed |
| `annual_fee_rate` | Annual percentage as decimal, e.g. 0.005 = 0.5% |
| `churn_90d` | Generated voluntary full-transfer outcome during next 90 days; target only |
| `outcome_window_end` | snapshot_date + 90 days; split maturity check, not feature |
| `actual_outflow_90d` | churn_90d × account_balance; post-outcome evaluation only |

## Generated relationships

A logistic formula raises simulated exit likelihood with contribution inactivity, complaints, transfer inquiry, noncontributing employment and higher fees, and lowers it with engagement, satisfaction and tenure. A small year effect creates temporal drift. The formula is illustrative, specified before training and not industry-estimated. Outcomes are Bernoulli draws; the latent probability is not saved as a feature. Satisfaction missingness is applied after outcome generation. Balances are lognormal with tenure scaling and bounds. All IDs are distinct and snapshots represent different members.

## Model output fields

| Field | Meaning |
|---|---|
| `churn_probability_90d` | Estimated voluntary departure risk within next 90 days |
| `churn_flag` | Probability at or above validation-selected threshold |
| `amount_if_member_leaves` | Balance × assumed transfer fraction |
| `expected_outflow_90d` | Probability × conditional transfer amount |
| `annual_fee_revenue_if_member_leaves` | Conditional transfer amount × annual fee rate |
| `expected_annual_fee_revenue_at_risk` | Probability-weighted annualized fee run-rate exposure |
| `expected_assets_retained` | Outflow × assumed conditional save rate |
| `expected_fee_revenue_preserved` | Annual fee exposure × save rate × fee years |
| `expected_net_contribution` | Preserved fees × margin − contact cost |

`actual_churn_90d` in evaluation exports corresponds to input target `churn_90d`. Inference exports contain no realized outcomes. All financial assumptions are explicit in the dashboard and business functions.
