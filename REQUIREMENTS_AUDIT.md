# Final requirements audit

## Verdict

The final main project follows the supplied “Customer Churn with Revenue Prioritization” breakdown and both final screenshots. The earlier pension-only edition did not fully match this final brief; it used synthetic records and annual fees. The primary project is now the public-Telco, monthly-value implementation. The synthetic pension work is retained separately to satisfy the earlier member/withdrawal task.

| Requirement | Final implementation | Evidence |
|---|---|---|
| Exact project title | Value-Based Customer Churn Prediction & Prioritized Retention List | README and dashboard |
| Public telecom/subscription dataset | Bundled public IBM Telco sample, upstream license/source retained | data/telco.csv, NOTICE, DATA_LICENSE |
| Simple Logistic Regression or Random Forest | One logistic regression plus prior baseline; no complex-model competition | src/train.py |
| Do not prioritize the model | Business decision, financial definitions and action list lead the project | README, CASE_STUDY.md |
| Recall favored over precision | Highest validation threshold attaining ≥85% recall; precision/volume burden reported | threshold tradeoff CSV, notebook |
| Weigh FP against FN | Explicit FP=$20 / FN=$100 assumption and ratios 1,3,5,10 | error_cost_sensitivity.csv, dashboard |
| P(churn) × monthly customer value | Exact expected_monthly_loss field and descending rank | prioritized_call_list.csv, src/business.py |
| Prioritized retention list | Full population ranking, flags and separate capacity/positive-value campaign queue | three output CSVs |
| Incentive/retention spend limit | Break-even total spend, post-outreach incentive cap and example support tiers | call list and dashboard |
| Resource allocation and ROI tradeoffs | Capacity, margin, horizon, cost, save-rate and tier controls | app.py, sensitivity.csv |
| Meaningful code comments | Explain leakage, recall rule, ranking separation, affordability and assumptions | src/train.py, src/business.py, src/score.py |
| Framework, approach and justified decisions | Business-first narrative and executed notebook | README, CASE_STUDY.md, notebook |
| Business recommendations | Ranked pilot, affordable interventions, cheaper effective treatment, randomized evaluation and monitoring | README recommendations |
| Recruiter/GitHub presentation | Executed notebook, charts, guide, tests, license, CI and reproduction workflows | repository root and .github/workflows |
| Earlier pension member/amount tasks | Separate synthetic extension with member flags, conditional balances and fee exposure | extensions/pension/README.md |

## Evidence boundaries

Recall target is set on validation; test recall is 82.9%, not guaranteed 85%. The sample is public sample data, not verified live company data. No prospective label horizon or causal retention result is established. Error penalties, save rates, retained months, margins and tier costs are assumptions. Offer tiers illustrate affordability, not optimal treatment. Financial estimates must not be described as actual money saved.

GitHub workflows are prepared; local verification does not imply they have run remotely. No external repository or hosted app was created. See VALIDATION.md for checks actually performed.
