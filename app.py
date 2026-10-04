"""Interactive customer-retention decision dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import intervention_business_case, lift_table, make_demo_accounts, root_cause_summary, threshold_curve, train_and_score, validate_accounts


st.set_page_config(page_title="Retention Command Center", page_icon="🧭", layout="wide", initial_sidebar_state="collapsed")
st.title("Retention Command Center")
st.caption("Churn risk, explainable drivers, targeting economics, and an account action queue · seeded demonstration data")

uploaded = st.sidebar.file_uploader("Upload account features", type="csv")
threshold = st.sidebar.slider("Intervention threshold", 0.30, 0.90, 0.70, 0.05)
save_rate = st.sidebar.slider("Expected save rate", 0.05, 0.40, 0.18, 0.01)
discount_rate = st.sidebar.slider("Three-month discount rate", 0.00, 0.30, 0.10, 0.01)
accounts = pd.read_csv(uploaded) if uploaded else make_demo_accounts()
quality = validate_accounts(accounts)
scored, coefficients, auc = train_and_score(accounts)
scored["root_cause"] = scored.apply(root_cause_summary, axis=1)
test = scored.loc[scored["split"] == "test"]
case = intervention_business_case(test, threshold, save_rate, discount_rate)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Holdout ROC AUC", f"{auc:.3f}")
c2.metric("Accounts targeted", f"{case['accounts_targeted']:,}")
c3.metric("Expected revenue saved", f"${case['expected_revenue_saved']:,.0f}")
c4.metric("Expected net value", f"${case['net_value']:,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Holdout lift by risk decile")
    lift = lift_table(scored)
    figure = px.bar(lift, x="risk_decile", y="lift", text_auto=".1f", color="lift", color_continuous_scale="Blues", labels={"risk_decile": "Risk decile (1 = highest)", "lift": "Churn lift vs. baseline"})
    figure.add_hline(y=1, line_dash="dash")
    st.plotly_chart(figure, width="stretch")
with right:
    st.subheader("Threshold tradeoff")
    curve = threshold_curve(scored)
    melted = curve.melt(id_vars="threshold", value_vars=["precision", "recall"], var_name="metric", value_name="value")
    figure = px.line(melted, x="threshold", y="value", color="metric", markers=True)
    figure.add_vline(x=threshold, line_dash="dash", line_color="#DC2626")
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(figure, width="stretch")

st.subheader("Priority account queue")
queue = test.loc[test["churn_probability"] >= threshold].sort_values(["churn_probability", "mrr"], ascending=False)
st.dataframe(queue[["account_id", "plan", "mrr", "churn_probability", "weekly_sessions", "tickets_30d", "nps", "root_cause"]], hide_index=True, width="stretch")

with st.expander("Model governance", expanded=True):
    coefficient_table = pd.DataFrame({"feature": coefficients.keys(), "coefficient": coefficients.values()}).sort_values("coefficient")
    st.plotly_chart(px.bar(coefficient_table, x="coefficient", y="feature", orientation="h", color="coefficient", color_continuous_scale="RdBu"), width="stretch")
    st.info("Risk scores prioritize review; they do not authorize automated customer treatment. Validate calibration, subgroup performance, offer eligibility, and intervention incrementality before production use.")

st.download_button("Download priority queue", queue.to_csv(index=False), "priority_accounts.csv", "text/csv")
