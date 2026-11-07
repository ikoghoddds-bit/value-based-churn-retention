# Model card — final aligned edition

## Intended use

Portfolio demonstration on public IBM Telco sample data. Simple logistic regression estimates the supplied churn label. It supports a recall-first screen, p × monthly value prioritization and transparent retention-spend scenarios. Not validated as a live forward-risk system.

## Data and evaluation

7,043 records, unique IDs, stratified 60/20/20 seed-42 split. 4,225 train, 1,409 validation, 1,409 test. Eleven blank TotalCharges values become missing values; training-fold medians impute them. Numeric scaling and categorical encoding remain inside the five calibration folds. IDs, target, gender, SeniorCitizen, Partner and Dependents are excluded from predictors. Demographics retained separately for descriptive audit.

Models: prior baseline and one logistic regression, C=1. Five-fold training-only sigmoid calibration. No model search. Threshold 0.24 is the highest validation threshold attaining ≥85% recall (achieved 85.3%). A separate cost-minimizing threshold is calculated on validation per assumed error-cost ratio for comparison; it does not replace the recall-first rule.

## Performance

Test recall 0.8289, precision 0.4844, F2 0.7257, ROC AUC 0.8327, average precision 0.6252, Brier 0.1413. Confusion matrix: [[705, 330], [64, 310]] (actual rows and predicted columns: stay/churn). AUC fixed-model bootstrap interval 0.8102–0.8541. Validation comparison metrics use 0.50; final test metrics use the recall-first threshold. Threshold changes affect flags, not probability × monthly value ranking.

## Financial assumptions and limits

Monthly charges proxy monthly customer revenue, not profit. Probability times monthly value is an exposure score; retained months is a valuation scenario, not an estimated survival horizon. Uniform save rate, margin, billing and costs are assumed; no causal retention effects, repeat churn, discounted lifetime value or realized gains are established. Incentive caps are affordability limits. Tier costs are illustrative; treatment-specific effectiveness is unknown. The uniform-cost campaign and tier scenario must not be double-counted.

## Generalization and controls

The cross-sectional sample lacks timestamps and intervention outcomes; pre-churn feature availability and a prospective target must be verified with real authorized data before deployment. Synthetic pension extension is separate. Do not combine its metrics or asset balances with the main Telco revenue figures. No real identities or employer data are added.

Batch scoring validates schema, unique IDs, known categories and finite nonnegative amounts; TotalCharges may be blank and imputed. Age/gender exclusion cannot eliminate proxies or guarantee fairness. Serialized models must be trusted. Dashboard runs locally; no external account integration, live retention action or production security review is included.

## Validation needed for real use

Time-stamped authorized customer snapshots, explicit forward label, point-in-time feature lineage, temporal holdout, calibrated risk, business-approved error costs, opt-in/contact governance and randomized offer-versus-control pilot. Measure incremental contribution with uncertainty. Monitor recall, volume, calibration, subgroup errors, feature drift and actual cost/return with versioned rollback criteria.
