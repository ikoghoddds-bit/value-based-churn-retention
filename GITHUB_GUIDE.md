# Publish the final portfolio on GitHub

Suggested repository: `value-based-churn-retention`  
Description: “Recall-focused customer churn prediction with probability-weighted monthly revenue prioritization, incentive spending limits, and an interactive retention dashboard.”  
Topics: `python`, `machine-learning`, `churn-prediction`, `logistic-regression`, `streamlit`, `customer-retention`, `portfolio-project`.

## Upload

Create an empty repository. Extract the ZIP and upload the **contents of `value-based-churn` at the repository root**, including `.github`, `.gitignore`, data, outputs and executed notebooks. Do not upload only the ZIP or nest the main project below the repository root. Root placement allows the supplied workflows to find requirements and tests.

```bash
git init
git add .
git commit -m "Add value-based churn and retention portfolio"
git branch -M main
# Replace YOUR_USERNAME:
git remote add origin https://github.com/YOUR_USERNAME/value-based-churn-retention.git
git push -u origin main
```

Use the normal workflow for an existing repository rather than reinitializing it. No remote has been created or pushed from this environment.

## Recruiter walkthrough

1. Read the README's business decision, exact monthly ranking formula and recall tradeoff.
2. Inspect the prioritized call list: probabilities, dollar exposure, break-even spend and support tiers.
3. Review the executed notebook, threshold/cost sensitivity and code comments explaining why choices matter.
4. Run the dashboard and vary save rate, margins, contact capacity and incentive costs.
5. Explain how a randomized pilot would establish actual retention ROI; modeled estimates are not achieved savings.

The public-data Telco project is the main result. The pension example is a separate synthetic domain extension. Avoid leading recruiters to believe the pension data are real.

## Checks and presentation

GitHub renders README, notebook outputs and PNG charts. Streamlit requires a running Python server; GitHub does not execute it. HTML overview can be downloaded and opened locally. Do not add a live-demo URL or passing badge until the respective deployment/Actions run succeeds.

The push/PR workflow runs main tests and example inference. The manual reproduction workflow reruns training and uploads regenerated evidence. Workflows have been supplied but not executed remotely here. For pension checks, run `python -m pytest -q` from `extensions/pension` in a separate process/environment context.

Keep the MIT code license, IBM sample-data license and attribution. Exclude caches/virtual environments using `.gitignore`; retain the trusted model and published evidence for a reviewer-friendly first run.
