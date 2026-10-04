# Methodology

## Modeling objective

Rank active accounts by near-term churn risk, explain observable risk signals, and evaluate whether a retention intervention can create positive expected value.

## Workflow

1. Validate one row per account and required behavioral, support, sentiment, tenure, plan, MRR, and outcome fields.
2. Standardize numeric features and one-hot encode plan.
3. Fit a regularized logistic model on a deterministic 70% training split.
4. Evaluate ROC AUC and lift on the untouched 30% holdout split.
5. Produce threshold precision/recall and intervention economics.
6. Create an account queue with deterministic, auditable driver summaries.

## Business case

`net value = annual revenue at risk × expected save rate − three-month intervention discount cost`

The save rate is a scenario assumption, not an observed causal effect. A production program should estimate incremental saves with a randomized holdout.

## Governance

Do not use protected characteristics or sensitive support text directly. Review subgroup calibration and error rates, monitor drift, document overrides, and require human approval before outreach.
