# Model card

## Identity and intended use

Version 2.0 · Pension synthetic demonstration · Joseph Ikogho. Educational forecasting and economic prioritization, not a validated pension advisory or production customer system. Goal: estimate voluntary full transfer within 90 days after snapshot.

## Data and provenance

14,400 generated members, 400 distinct members per month January 2023–December 2025, one observation per member. Random seed 2026; generator is versioned. Balances, demographics, activity and outcomes are fabricated; relationships are specified in code and not industry-estimated. Survey responses have 6% missingness. SHA-256: `9dcf3426c1634a6dd30ef1cdf51ae1ab9681bd0f84e4d9762fa7531d394d30f9`. No real people, employer or fund data are used.

## Method and evaluation

Base fit 4,800, calibration 1,200, validation 2,400 and test 2,400. Three 1,200-row label-maturity embargoes exclude 3,600 rows in total. Numeric imputation/scaling and categorical encoding fit on base-fit rows only. Frozen candidate models are sigmoid calibrated on later rows. Validation Brier selects the model; validation F2 selects threshold 0.10; no test refitting or tuning.

Test ROC AUC 0.7747; average precision 0.3316; Brier 0.0851; recall 0.7077; precision 0.2280. Confusion matrix (actual rows, predicted columns: stay/leave): [[1517, 623], [76, 184]]. AUC 95% interval 0.7411–0.8042 uses 1,000 fixed-model test-row resamples. Comparison table precision/recall use 0.50, while final metrics use selected threshold.

## Financial limits

Full transfer of snapshot balance by definition; no partial transfer modeling, returns, benefit entitlements, fees beyond a simple percentage, future contributions or tax treatment. A lower transfer fraction is sensitivity only. Exposure sums pool six different snapshot cohorts, not one present-day fund. Annual fee exposure is a run-rate after exit, not a 90-day loss. Constant-rate fee-year valuation ignores timing, discounting and repeat exit. Save rate is assumed uniform and causal impact is unknown. Member assets are not fee revenue.

## Fairness, use and controls

Age is excluded; descriptive age-group audit does not establish real-world fairness. Employment and activity may act as proxies. A flag is not proof of intent and must not restrict transfer rights. No protected-attribute parity guarantee, human outreach integration or production access management is provided. Input validator rejects missing schema, duplicate members, invalid categories, out-of-range fees and nonfinite core amounts. Model serialization must be trusted. Pipeline can impute missing survey scores; production should monitor that missingness.

## Deployment criteria

Fund-specific outcome definitions and transferable balances, authorized de-identified observations, verified pre-snapshot windows, common-cutoff scoring roster, temporal validation and real calibration. Measure offer-vs-control incremental outcomes with prespecified costs and horizon before claiming retention gains. Monitor data drift, schema errors, calibration, PR performance, subgroup errors and actual economic return. Define thresholds and rollback criteria before pilot launch.
