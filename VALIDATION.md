# Final validation record

November 6, 2026 · Python 3.12 · Public Telco main project, synthetic pension extension

## Main project

- Compared final brief and both screenshots against code, data, outputs and narrative; mapping in REQUIREMENTS_AUDIT.md.
- Trained one logistic regression plus prior baseline with preprocessing inside five training-only calibration folds.
- Recomputed validation threshold table, selected highest threshold satisfying 85% validation recall, and evaluated frozen choices on the held-out test set.
- Recomputed exact probability × monthly-value ranking, flagged subset, spending caps, example tiers and capacity-constrained campaign.
- Executed all nine notebook code cells sequentially with outputs and figures. In-process IPython was used because this build environment blocks kernel sockets; the delivered notebook is standard Python 3.
- Eight tests passed, covering campaign arithmetic/capacity, unique split membership, scoring schema, saved-model consistency, exact monthly ranking and affordability, recall-rule/export consistency, invalid numeric/ID input and Streamlit rendering/zero-save behavior.
- Batch inference succeeded on the target-free example CSV.
- Evaluation chart visually checked for clipping and legibility.
- Verified actionable CSVs exclude target labels, sorting matches the exact score and notebook outputs contain no errors.

## Pension extension

Five existing extension tests passed in a separate process, covering economics, temporal leakage controls, schema, serialization and dashboard zero-effect cases. Its metrics and synthetic labels remain separate from the public-data main project.

## Publication and interpretation

Local verification does not establish real campaign effectiveness or prospective performance. FP/FN costs, margins, save rate, support costs and retained horizons are assumptions. GitHub workflows are supplied but were not executed remotely. No remote repository or hosted app was created. Archive structure and notebook content were checked before saving.
