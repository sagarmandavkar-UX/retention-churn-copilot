"""Retention cohorts, churn scoring, explanations, and intervention economics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["weekly_sessions", "tickets_30d", "nps", "tenure_months", "plan"]


def make_demo_accounts(seed: int = 31, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.55, 0.35, 0.10])
    sessions = rng.gamma(2.2, 2.5, n)
    tickets = rng.poisson(1.4, n)
    nps = np.clip(rng.normal(35, 28, n), -100, 100)
    tenure = rng.integers(1, 48, n)
    mrr = np.select([plan == "basic", plan == "pro"], [49, 149], default=799).astype(float)
    logit = 1.0 - 0.32 * sessions + 0.48 * tickets - 0.018 * nps - 0.025 * tenure + (plan == "basic") * 0.35
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)
    return pd.DataFrame({
        "account_id": [f"acct_{i:05d}" for i in range(n)], "weekly_sessions": sessions,
        "tickets_30d": tickets, "nps": nps, "tenure_months": tenure, "plan": plan,
        "mrr": mrr, "churned": churned,
    })


def _design_matrix(accounts: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = ["weekly_sessions", "tickets_30d", "nps", "tenure_months"]
    standardized = (accounts[numeric] - accounts[numeric].mean()) / accounts[numeric].std()
    plans = pd.get_dummies(accounts["plan"], prefix="plan", drop_first=True, dtype=float)
    names = ["intercept", *numeric, *plans.columns]
    return np.column_stack([np.ones(len(accounts)), standardized.to_numpy(), plans.to_numpy()]), names


def train_and_score(accounts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], float]:
    validate_accounts(accounts)
    x, names = _design_matrix(accounts)
    y = accounts["churned"].to_numpy(dtype=float)
    rng = np.random.default_rng(2026)
    train_mask = rng.random(len(accounts)) < 0.70
    beta = np.zeros(x.shape[1])
    for _ in range(40):
        probability = 1 / (1 + np.exp(-np.clip(x[train_mask] @ beta, -30, 30)))
        weights = np.clip(probability * (1 - probability), 1e-6, None)
        hessian = x[train_mask].T @ (weights[:, None] * x[train_mask]) + 1e-4 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, x[train_mask].T @ (y[train_mask] - probability))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    scored = accounts.copy()
    scored["churn_probability"] = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    scored["split"] = np.where(train_mask, "train", "test")
    test = scored.loc[~train_mask]
    ranks = test["churn_probability"].rank(method="average")
    test_y = test["churned"].to_numpy(dtype=float)
    positives, negatives = int(test_y.sum()), int(len(test_y) - test_y.sum())
    auc = (float(ranks[test_y == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    return scored, dict(zip(names, beta.astype(float))), float(auc)


def validate_accounts(accounts: pd.DataFrame) -> dict[str, int]:
    required = {"account_id", *FEATURES, "mrr", "churned"}
    missing = required.difference(accounts.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    duplicates = int(accounts["account_id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate accounts")
    return {"rows": len(accounts), "accounts": int(accounts["account_id"].nunique()), "churned": int(accounts["churned"].sum())}


def lift_table(scored: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Rank the test population into risk deciles and calculate lift."""
    data = scored.loc[scored["split"] == "test"].sort_values("churn_probability", ascending=False).copy()
    data["risk_decile"] = pd.qcut(np.arange(len(data)), bins, labels=range(1, bins + 1))
    baseline = data["churned"].mean()
    table = data.groupby("risk_decile", observed=True).agg(accounts=("account_id", "size"), churn_rate=("churned", "mean"), avg_score=("churn_probability", "mean"), mrr=("mrr", "sum")).reset_index()
    table["lift"] = table["churn_rate"] / baseline
    return table


def threshold_curve(scored: pd.DataFrame, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    thresholds = np.asarray(thresholds if thresholds is not None else np.arange(0.30, 0.91, 0.05))
    test = scored.loc[scored["split"] == "test"]
    rows = []
    for threshold in thresholds:
        predicted = test["churn_probability"] >= threshold
        tp = int(((test["churned"] == 1) & predicted).sum())
        fp = int(((test["churned"] == 0) & predicted).sum())
        fn = int(((test["churned"] == 1) & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        business = intervention_business_case(test, float(threshold))
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, **business})
    return pd.DataFrame(rows)


def root_cause_summary(account: pd.Series) -> str:
    drivers = []
    if account["weekly_sessions"] < 3:
        drivers.append("low weekly engagement")
    if account["tickets_30d"] >= 3:
        drivers.append("elevated support volume")
    if account["nps"] < 10:
        drivers.append("low NPS")
    return "Primary signals: " + (", ".join(drivers) if drivers else "no single dominant behavioral signal") + "."


def intervention_business_case(scored: pd.DataFrame, threshold: float = 0.70, save_rate: float = 0.18, discount_rate: float = 0.10) -> dict[str, float]:
    targeted = scored.loc[scored["churn_probability"] >= threshold]
    annual_revenue_at_risk = float((targeted["mrr"] * 12).sum())
    saved = annual_revenue_at_risk * save_rate
    cost = float((targeted["mrr"] * discount_rate * 3).sum())
    return {"accounts_targeted": len(targeted), "annual_revenue_at_risk": annual_revenue_at_risk, "expected_revenue_saved": saved, "intervention_cost": cost, "net_value": saved - cost}


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    accounts = make_demo_accounts()
    quality = validate_accounts(accounts)
    scored, coefficients, auc = train_and_score(accounts)
    scored["root_cause"] = scored.apply(root_cause_summary, axis=1)
    scored.nlargest(100, "churn_probability").to_csv(out / "priority_accounts.csv", index=False)
    lift_table(scored).to_csv(out / "lift_table.csv", index=False)
    threshold_curve(scored).to_csv(out / "threshold_curve.csv", index=False)
    result = {"data_quality": quality, "test_roc_auc": auc, "coefficients": coefficients, "threshold_70pct": intervention_business_case(scored.loc[scored["split"] == "test"])}
    (out / "model_card.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
