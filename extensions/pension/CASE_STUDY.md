# Pension retention: identify potential leavers and attach the money

Joseph Ikogho · Data Science Portfolio Project 1 · Synthetic demonstration

## Business problem

A pension administrator wants to identify members likely to leave and understand the amount at risk for each. A raw prediction is insufficient: losing a large account removes assets from the fund, while the administrator loses only the associated fees. The project makes that distinction explicit.

## What I built

A reproducible Python pipeline, executed notebook and local Streamlit dashboard on 14,400 explicitly synthetic member snapshots. Each member receives a 90-day departure probability, a risk flag, a conditional transfer amount, a probability-weighted outflow estimate and an annualized fee exposure. A separate queue prioritizes positive-value outreach under capacity and campaign-cost assumptions.

## Evaluation and result

An embargoed temporal evaluation separates base fitting, probability calibration, validation and test cohorts. A prior baseline, logistic regression and gradient boosting are compared; validation Brier selects logistic regression and F2 selects a 0.10 threshold. Test ROC AUC is 0.775, average precision 0.332 and recall 70.8%. Precision is 22.8%; the model catches 184 of 260 synthetic departures but also flags 623 members who stay.

For the pooled 2,400-member test cohorts, expected outflow is $19,294,569 in assets and annualized fee exposure is $114,992. These are distinct business quantities. The default scenario selects 142 of at most 200 contacts and estimates $11,023 preserved fee revenue and $4,166 net contribution. A 25% save rate, one fee year, 70% margin and $25/contact are assumed—not measured.

## How this addresses the two tasks

1. `flagged_members.csv` identifies members with probability above the selected threshold.
2. The same file lists how much each could take if they leave and the expected amount after weighting by probability. `member_predictions.csv` covers the full holdout so unflagged exposure remains visible.

## What I can defend in an interview

**Why F2?** The requested goal emphasizes finding leavers. F2 weights recall more than precision; the false-positive burden is reported openly. Economic contact selection remains separate.

**Why calibration and Brier?** Dollars depend on the probability scale, not just ranking. Calibration uses later observations than fitting, with no overlap in outcome maturity.

**Why a time split?** The intended use predicts future departures. Ninety-day embargoes prevent training or selection from using outcomes that would not have been known at the next stage.

**Why synthetic data?** No pension records were supplied. A transparent generator permits reproducibility without suggesting telecom customers are pension members. Results prove implementation behavior, not pension-industry performance.

**How much does the business lose?** Assets outflow and fee loss must not be combined. A transferred $100,000 balance at a 0.5% fee represents $500 annualized fee run-rate loss conditional on exit—not $100,000 revenue loss.

**Did the project stop members leaving?** No. That requires a real intervention and control group. The project supports targeting and evaluation design.

**What changes for production?** Real transfer definitions, authorization and de-identification, point-in-time feature verification, current eligible roster, temporal external validation, contact controls, outcome monitoring and a randomized retention pilot.

## Portfolio description

“Built a pension-member churn and financial-exposure portfolio application using 14,400 synthetic snapshots, embargoed temporal validation and calibrated classification; delivered member-level departure flags, asset-outflow estimates, fee-revenue exposure and a cost-aware retention queue.”

Do not describe simulated dollar estimates as achieved savings or the model as validated on real pension members.
