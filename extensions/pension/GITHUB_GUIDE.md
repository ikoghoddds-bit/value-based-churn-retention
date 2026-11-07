# Publish this portfolio on GitHub

## Suggested repository details

Name: `pension-churn-financial-exposure`  
Description: “Synthetic pension-member churn prediction with temporal validation, member-level outflow estimates, fee-revenue exposure, and a Streamlit retention dashboard.”  
Topics: `machine-learning`, `churn-prediction`, `python`, `scikit-learn`, `streamlit`, `synthetic-data`, `portfolio-project`, `customer-retention`.

## Upload the project

Create an empty GitHub repository. Extract the package and upload the **contents** of `pension-churn-revenue` at the repository root, including `.github`, `.gitignore`, the executed notebook and outputs. Do not upload only the ZIP. Avoid nesting the project one level below the repository root; the workflows expect `requirements.txt` at the root.

Using Git from a terminal inside the extracted project directory:

```bash
git init
git add .
git commit -m "Add pension churn and financial exposure portfolio"
git branch -M main
# Replace YOUR_USERNAME with your GitHub username:
git remote add origin https://github.com/YOUR_USERNAME/pension-churn-financial-exposure.git
git push -u origin main
```

If your repository already has history, use its existing checkout and normal commit workflow instead. No credentials are included in this project. The `.gitignore` excludes virtual environments and caches.

## Recruiter walkthrough

1. Lead with the two task outputs in README: who might leave and how much each could take.
2. Show the diagnostic chart and honest precision/recall tradeoff.
3. Open the executed notebook for data generation, evaluation and financial reasoning.
4. Run the dashboard to change the risk threshold, transfer fraction and campaign economics.
5. Distinguish synthetic test evidence from real business results. Explain how a randomized pilot would validate retention effect.

GitHub renders the README, notebook and PNG directly. GitHub does not execute Streamlit or render the standalone HTML as a live app; download/open the HTML locally. The dashboard runs locally using the README commands. You can host it separately later; no live-demo URL or passing CI badge is invented here.

## Automated checks

`Portfolio checks` runs pytest and example CSV scoring on pushes and pull requests. `Reproduce training` is manually triggered in the Actions tab and uploads regenerated outputs. These workflows are supplied but have not been run on GitHub from this environment. Review the first Actions run after uploading.

## Suggested recruiter-safe summary

“Developed an end-to-end pension-member churn demonstration on synthetic data, connecting calibrated 90-day risk to individual transfer exposure, annualized fee risk and retention investment decisions.”

Keep “synthetic” visible in the repository, LinkedIn description and interviews. Do not claim real customers retained or revenue saved.
