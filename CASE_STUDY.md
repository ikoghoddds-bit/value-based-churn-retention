# Value-Based Customer Churn Prediction & Prioritized Retention List

Joseph Ikogho · Data Science Portfolio Project 1

## Business problem and framework

Retention teams need more than a churn label. Missing a valuable churner can be expensive, but contacting everyone wastes capacity. The framework separates three decisions: identify likely churners with a recall-focused screen; rank financial exposure using probability × monthly value; and limit retention spend to the contribution plausibly preserved.

## Approach and choices

I used the public IBM Telco sample and a simple calibrated logistic regression with a prior baseline. A 60/20/20 stratified split separates training, validation and test customers. Preprocessing stays inside training calibration folds. Simplicity makes the focus business allocation rather than model competition.

The threshold is the highest value achieving 85% validation recall, rather than the standard 0.50. The model is then frozen for test evaluation. I explicitly compare false-positive and false-negative penalties under several ratios; these are assumptions, not measured company economics.

## Measured model results

Held-out recall is 82.9%, versus 53.5% at 0.50. The recall-first rule catches 310 churners, misses 64 and flags 330 staying customers. Precision is 48.4%. AUC is 0.833 and average precision 0.625. The higher call burden is a deliberate, reported tradeoff.

## Financial and operational result

The action list ranks all 1,409 test customers by p × monthly charges and attaches a break-even retention budget, incentive cap and illustrative affordable support tier. Total monthly exposure is $28,314. Under a separate uniform $20 all-in campaign scenario with 25% save rate, six retained months and 70% margin, the top 200 positive-value contacts yield $17,781 expected saved revenue and $8,447 expected net contribution. These are modeled scenarios, not achieved outcomes.

## Recommendations

Use the highest monthly exposures for a budget-controlled randomized retention pilot. Keep low-value flags from consuming the whole budget. Treat the incentive cap as an upper bound, choose the least expensive proven-effective treatment, and record actual outreach and redemption costs. Measure incremental retained contribution against control before making savings claims. Use real cost ratios to revisit screening and monitor probability calibration and subgroup errors.

## Interview questions I can answer

**Why recall?** The brief makes a missed churner more costly than an unnecessary call. A validation recall target makes that preference concrete while avoiding the meaningless all-positive maximum-recall rule.

**Why not just rank risk?** A $100/month customer at 50% risk has $50 monthly exposure; a $20/month customer at 90% risk has $18. The first account has greater financial exposure despite lower probability.

**Why calibration?** The probability scale directly determines dollars. AUC assesses ranking, not whether 50% risk is actually near 50%.

**How much incentive is justified?** At most p × monthly value × retained months × assumed save rate × margin, less outreach cost. Staying strictly below that amount preserves modeled contribution.

**Are support tiers optimal?** No. They demonstrate affordability. Their effects are not learned; offer-specific causal effectiveness requires experimental evidence.

**Did this save the business money?** No measured retention campaign was run. The project connects predictions to a testable financial decision and shows what must be validated.

**Why a pension extension?** Earlier requirements used member transfers. The main project now exactly follows the public-Telco/monthly-value walkthrough, while the separate synthetic extension preserves member-level transfer analysis without mislabeling assets as business revenue.

## Recruiter-safe description

“Built a calibrated churn and revenue-prioritization portfolio application using a public Telco dataset; achieved 82.9% held-out recall, ranked customers by probability-weighted monthly value, and produced cost-aware retention and incentive scenarios.”
