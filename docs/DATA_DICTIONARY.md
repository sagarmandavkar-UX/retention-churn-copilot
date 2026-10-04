# Data dictionary

| Field | Definition |
|---|---|
| `account_id` | Unique account identifier |
| `weekly_sessions` | Recent weekly product engagement |
| `tickets_30d` | Support tickets opened in the prior 30 days |
| `nps` | Latest Net Promoter Score |
| `tenure_months` | Months since account activation |
| `plan` | Commercial plan category |
| `mrr` | Monthly recurring revenue |
| `churned` | Observed binary churn outcome |
| `churn_probability` | Model-estimated risk |
| `root_cause` | Deterministic summary of observed behavioral risk signals |

The bundled account table is synthetic. Production definitions must specify the prediction timestamp, label window, inactivity rules, cancellation timing, and eligibility population.
