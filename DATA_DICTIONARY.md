# Data dictionary

IBM sample; units as interpreted by the Telco schema. No timestamp or treatment columns are present.

| Column | Meaning | Values/type |
|---|---|---|
| customerID | Unique customer identifier; excluded from prediction | String |
| gender | Reported gender; audit only | Female, Male |
| SeniorCitizen | Senior indicator (0/1); audit only | 0, 1 |
| Partner | Partner indicator; excluded | Yes, No |
| Dependents | Dependent indicator; excluded | No, Yes |
| tenure | Customer tenure in months | Numeric |
| PhoneService | Phone service subscription | No, Yes |
| MultipleLines | Multiple phone lines status | No phone service, No, Yes |
| InternetService | Internet service type | DSL, Fiber optic, No |
| OnlineSecurity | Online security service status | No, Yes, No internet service |
| OnlineBackup | Online backup service status | Yes, No, No internet service |
| DeviceProtection | Device protection service status | No, Yes, No internet service |
| TechSupport | Technical support service status | No, Yes, No internet service |
| StreamingTV | Streaming TV service status | No, Yes, No internet service |
| StreamingMovies | Streaming movies service status | No, Yes, No internet service |
| Contract | Subscription contract term | Month-to-month, One year, Two year |
| PaperlessBilling | Paperless billing indicator | Yes, No |
| PaymentMethod | Payment method category | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| MonthlyCharges | Monthly charge in USD; used for revenue valuation | Numeric |
| TotalCharges | Cumulative billed charges in USD; blanks imputed | Numeric |
| Churn | Binary observed churn label, Yes/No; target only | No, Yes |

## Main decision outputs

| Field | Definition |
|---|---|
| priority_rank | Descending expected_monthly_loss; ID breaks ties |
| probability | Calibrated churn-label probability |
| churn_flag | Probability ≥ validation recall threshold |
| monthly_customer_value | MonthlyCharges, a revenue proxy |
| expected_monthly_loss | probability × monthly_customer_value |
| max_total_retention_spend | p × value × retained months × save rate × margin |
| max_incentive_after_outreach | max(0, max_total_retention_spend − outreach cost) |
| suggested_support_tier | Highest illustrative tier strictly below the affordable cap |
| planned_total_cost | Outreach plus chosen tier cost, or zero for no paid outreach |
| expected_net_after_suggested_tier | Expected contribution benefit less that tier's total cost |

The main dataset has no timestamps or forward observation windows. Monthly value is a billing unit, not evidence that the predicted churn label is a prospective one-month outcome. The pension extension has its own dictionary and synthetic provenance; its fields and results must not be mixed with these.
