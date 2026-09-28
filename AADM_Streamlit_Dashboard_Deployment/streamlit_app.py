"""Quick commerce analytics dashboard for Streamlit Community Cloud.

This read-only app uses anonymized order-level fields exported by analysis.py.
Supervised model metrics were computed once on a chronological holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
COLORS = {
    "navy": "#102b3a",
    "teal": "#087e82",
    "orange": "#c9633b",
    "gray": "#889aa5",
}

st.set_page_config(
    page_title="Quick Commerce Order Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 78rem; }
      h1, h2, h3 { color: #102b3a; letter-spacing: -.02em; }
      div[data-testid="stMetric"] { border: 1px solid #d9e5e8; border-top: 3px solid #087e82;
        background: #fff; padding: 1rem 1.15rem; border-radius: .65rem; }
      div[data-testid="stMetric"] label { color: #4f6571; }
      div[data-testid="stMetricValue"] { color: #102b3a; }
      .stAlert { border-radius: .65rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, dict]:
    data = pd.read_json(ROOT / "data" / "orders.json", orient="records")
    data["Order_Date"] = pd.to_datetime(data["Order_Date"], format="%Y-%m-%d")
    summary = json.loads((ROOT / "data" / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "Order_Date", "Platform", "City", "Product_Category", "Order_Status",
        "Stock_Status", "Final_Payment_INR", "Delay_Min", "Late_5_Min", "Cluster",
    }
    if not expected.issubset(data.columns) or len(data) != 2000:
        raise ValueError("The packaged order data does not match the analyzed sample.")
    return data, summary


def rupees_lakh(amount: float) -> str:
    return f"₹{amount / 100_000:,.2f} lakh"


def bar(data: pd.DataFrame, x: str, y: str, color: str, *, sort="-y", height=270):
    return (
        alt.Chart(data)
        .mark_bar(color=color, cornerRadiusEnd=4)
        .encode(
            x=alt.X(x, sort=sort, title=None),
            y=alt.Y(y, title=None),
            tooltip=[alt.Tooltip(x), alt.Tooltip(y)],
        )
        .properties(height=height)
    )


try:
    orders, summary = load_data()
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"The packaged data could not be loaded: {exc}")
    st.stop()

quality = summary["quality"]
overview = summary["overview"]
models = summary["models"]
split = summary["split"]

st.sidebar.header("Filter observed orders")
city = st.sidebar.selectbox("City", ["All cities", *sorted(orders.City.unique())])
platform = st.sidebar.selectbox("Platform", ["Both platforms", *sorted(orders.Platform.unique())])
category = st.sidebar.selectbox("Product category", ["All categories", *sorted(orders.Product_Category.unique())])
status = st.sidebar.selectbox("Order status", ["All outcomes", "Delivered", "Cancelled", "Returned"])
st.sidebar.caption("Filters affect the descriptive views. Model evaluation uses the fixed, later-date holdout.")

filtered = orders.copy()
if city != "All cities":
    filtered = filtered[filtered.City == city]
if platform != "Both platforms":
    filtered = filtered[filtered.Platform == platform]
if category != "All categories":
    filtered = filtered[filtered.Product_Category == category]
if status != "All outcomes":
    filtered = filtered[filtered.Order_Status == status]
delivered = filtered[filtered.Order_Status == "Delivered"]
late_count = int(delivered.Late_5_Min.sum())
cancel_count = int((filtered.Order_Status == "Cancelled").sum())

st.title("Quick commerce order analytics")
st.caption(
    "Advanced Analytics for Decision Making · Supplied sample: 2,000 orders, "
    "1 January–7 September 2026 · Anonymized deployment data"
)

overview_tab, eta_tab, models_tab, decisions_tab, quality_tab = st.tabs(
    ["Order overview", "Delivery and ETA", "Model evidence", "Decisions", "Data quality"]
)

with overview_tab:
    a, b, c, d = st.columns(4)
    a.metric("Selected orders", f"{len(filtered):,}")
    b.metric("Delivered payments recorded", rupees_lakh(delivered.Final_Payment_INR.sum()))
    c.metric("More than 5 min late", f"{late_count / len(delivered):.1%}" if len(delivered) else "—")
    d.metric("Cancelled", f"{cancel_count:,} ({cancel_count / len(filtered):.1%})" if len(filtered) else "0")
    st.caption(
        "Payment is the sum recorded on delivered rows, not verified recognized revenue or profit. "
        "Lateness uses delivered orders only."
    )
    if filtered.empty:
        st.info("No orders match these filters. Adjust the selections in the sidebar.")
    else:
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.subheader("Order outcomes")
            outcome = (
                filtered.groupby("Order_Status", as_index=False).size()
                .rename(columns={"size": "Orders"})
            )
            st.altair_chart(
                bar(outcome, "Order_Status:N", "Orders:Q", COLORS["teal"], sort=None),
                width="stretch",
            )
        with right:
            st.subheader("Late deliveries by city")
            city_late = (
                delivered.groupby("City", as_index=False)["Late_5_Min"].sum()
                .rename(columns={"Late_5_Min": "Late orders"})
            )
            if city_late.empty:
                st.info("Select delivered orders to see delivery timing.")
            else:
                st.altair_chart(
                    bar(city_late, "City:N", "Late orders:Q", COLORS["orange"]),
                    width="stretch",
                )
            st.caption("These are case counts, not a ranking adjusted for order volume.")

        monthly = (
            filtered.assign(Month=filtered.Order_Date.dt.to_period("M").astype(str))
            .groupby("Month", as_index=False)
            .agg(Orders=("Order_Status", "size"),
                 Cancelled=("Order_Status", lambda s: int((s == "Cancelled").sum())))
        )
        monthly["Cancellation rate"] = monthly.Cancelled / monthly.Orders
        st.subheader("Monthly recorded cancellation rate")
        st.altair_chart(
            alt.Chart(monthly).mark_line(point=True, color=COLORS["teal"], strokeWidth=3)
            .encode(x=alt.X("Month:N", title=None),
                    y=alt.Y("Cancellation rate:Q", title="Share of selected orders", axis=alt.Axis(format="%")),
                    tooltip=["Month:N", "Orders:Q", "Cancelled:Q", alt.Tooltip("Cancellation rate:Q", format=".1%")])
            .properties(height=220),
            width="stretch",
        )
        st.caption("There are exactly eight sampled orders per day in the source; this is not a population demand trend.")

with eta_tab:
    st.subheader("The ETA is systematically short in this sample")
    st.write(
        "Across all 1,780 delivered records, the mean error is **5.12 minutes**. "
        "On the later 354 delivered records, the original estimate has **6.62 minutes MAE**. "
        "Adding the **5.03-minute bias measured only in training** gives **5.31 minutes MAE**. "
        "The larger linear regression reaches 5.36 minutes MAE."
    )
    st.info("A more accurate promise does not make the physical delivery faster. Test customer effects before rollout.")

    holdout = orders[
        (orders.Order_Date >= pd.Timestamp(split["test_start"]))
        & (orders.Order_Status == "Delivered")
    ]
    if len(holdout) != models["linear"]["test_n"]:
        st.error("The packaged holdout does not reconcile with the report.")
        st.stop()
    correction = st.slider("Explore an ETA correction in minutes", 0.0, 10.0, 5.0, 0.5)
    scenario_mae = float((holdout.Delay_Min - correction).abs().mean())
    initial_mae = float(holdout.Delay_Min.abs().mean())
    a, b, c = st.columns(3)
    a.metric("Original ETA MAE", f"{initial_mae:.2f} min")
    b.metric("Scenario MAE", f"{scenario_mae:.2f} min", f"{initial_mae-scenario_mae:.2f} min improvement")
    c.metric("Training-derived correction", f"+{models['linear']['training_eta_bias_min']:.2f} min")
    grid = pd.DataFrame({"Correction": [i / 2 for i in range(21)]})
    grid["MAE"] = grid.Correction.map(lambda v: float((holdout.Delay_Min-v).abs().mean()))
    line = alt.Chart(grid).mark_line(point=True, color=COLORS["teal"], strokeWidth=3).encode(
        x=alt.X("Correction:Q", title="Minutes added to ETA"),
        y=alt.Y("MAE:Q", title="Holdout mean absolute error (minutes)"),
        tooltip=[alt.Tooltip("Correction:Q", format=".1f"), alt.Tooltip("MAE:Q", format=".2f")],
    ).properties(height=280)
    st.altair_chart(line, width="stretch")
    st.caption(
        "The slider inspects the already-used holdout; selecting its best point would overfit it. "
        "The recommended 5.03-minute correction was estimated from training dates alone."
    )
    st.write(
        f"Among the currently filtered delivered orders, **{late_count:,} of {len(delivered):,}** "
        "exceed the original ETA by more than five minutes."
    )

with models_tab:
    st.subheader("Five course techniques, one chronological evaluation")
    st.write(
        f"Training: **{split['train_all_n']:,} orders** through {split['train_end']}; "
        f"holdout: **{split['test_all_n']:,} orders** from {split['test_start']} to {split['test_end']}. "
        "Delivery models use delivered rows only. Model predictors exclude actual time, order status, "
        "final payment and post-order ratings."
    )
    model_table = pd.DataFrame([
        ["Linear regression", "Actual minutes", f"R² {models['linear']['r2_test']:.3f}; MAE {models['linear']['mae_test']:.2f} min", "Bias-corrected original ETA performs slightly better"],
        ["Logistic regression", ">5 min late", f"AUC {models['logistic']['roc_auc']:.3f}; recall {models['logistic']['recall']:.1%}", "Do not automate late-risk alerts"],
        ["K means", "Order profiles", f"k={models['cluster']['selected_k']}; silhouette {models['cluster']['silhouette_selected_train']:.3f}", "Weak descriptive grouping"],
        ["Decision tree", "Cancelled", f"AUC {models['classifiers']['decision_tree']['roc_auc']:.3f}; precision {models['classifiers']['decision_tree']['precision']:.1%}", "Do not score individual orders"],
        ["Random forest", "Cancelled", f"AUC {models['classifiers']['random_forest']['roc_auc']:.3f}; precision {models['classifiers']['random_forest']['precision']:.1%}", "Below the 92.5% no-cancellation accuracy baseline"],
    ], columns=["Technique", "Outcome", "Holdout result", "Decision"])
    st.dataframe(model_table, hide_index=True, width="stretch")
    st.warning(
        "ROC AUC near 0.5 means little ranking ability. The five order clusters have weak separation; "
        "they are not verified customer segments."
    )
    with st.expander("Confusion matrices and cluster profiles"):
        tree = models["classifiers"]["decision_tree"]["confusion_matrix_TN_FP_FN_TP"]
        forest = models["classifiers"]["random_forest"]["confusion_matrix_TN_FP_FN_TP"]
        logistic = models["logistic"]["confusion_matrix_TN_FP_FN_TP"]
        st.dataframe(
            pd.DataFrame([["Late logistic", *logistic], ["Cancellation tree", *tree], ["Cancellation forest", *forest]],
                         columns=["Model", "TN", "FP", "FN", "TP"]),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(pd.DataFrame(models["cluster"]["profiles"]), hide_index=True, width="stretch")

with decisions_tab:
    st.subheader("Recommendations with measurement gates")
    actions = [
        ("Pilot a +5 minute ETA display correction", "Operations", "Randomize the promise, then compare ETA error, complaints, conversion and actual delivery time. The holdout improvement is in forecast accuracy only."),
        ("Reconcile stock and fulfillment events", "Inventory and data engineering", "Investigate 144 rows labeled both Out of Stock and Delivered; check timestamps, substitution and replenishment."),
        ("Instrument the delivery lifecycle", "Dispatch and product", "Record allocation, pickup, route, rider capacity and checkout promise times before attempting new risk models."),
        ("Review the largest sample case pools", "Service operations", "Delhi and Chennai each have 117 late delivered rows. Use case volume for the first audit, with denominators retained."),
        ("Validate order economics before changing offers", "Finance and growth", "Join margin and actual promotion cost to the ₹43.47 lakh of delivered recorded payments and ₹9.45 lakh of listed discounts."),
        ("Retain simple baselines as release gates", "Analytics", "Require an independent later-date improvement in AUC, calibration and net benefit before automated cancellation or late-risk decisions."),
    ]
    for i, (title, owner, detail) in enumerate(actions, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {title}** · {owner}")
            st.write(detail)
    st.caption("No measured causal effect, recognized revenue or profit estimate is available in this dataset.")

with quality_tab:
    st.subheader("Data quality and boundaries")
    checks = pd.DataFrame([
        ["Rows / original columns", f"{quality['rows']:,} / {quality['columns_original']}", "One row is a recorded order"],
        ["Missing cells / duplicate IDs", f"{quality['missing_cells']} / {quality['duplicate_order_ids']}", "No records deleted or imputed in source"],
        ["Orders on each date", str(quality["orders_per_day_unique_values"]), "Fixed daily sample; do not forecast population demand"],
        ["Delivered while Out of Stock", str(quality["delivered_while_out_of_stock"]), "Requires source-event reconciliation"],
        ["Non-delivered with actual minutes", str(quality["non_delivered_with_actual_minutes"]), "Excluded from delivery timing analysis"],
        ["IQR flags retained", f"{quality['iqr_flags_kept']['Order_Value_INR']} order values; {quality['iqr_flags_kept']['Final_Payment_INR']} payments", "Plausible large baskets were kept"],
    ], columns=["Check", "Result", "Interpretation"])
    st.dataframe(checks, hide_index=True, width="stretch")
    st.write(
        "The public app contains only the fields needed for these views. Customer IDs and order IDs are not deployed. "
        "The order sample appears curated, and the timing of some predictors is undocumented. "
        "The original report and notebook contain the complete analysis and definitions."
    )
    st.subheader("Inspect anonymized records")
    display_cols = ["Order_Date", "Platform", "City", "Product_Category", "Order_Status",
                    "Stock_Status", "Final_Payment_INR", "Delay_Min", "Cluster"]
    st.dataframe(filtered[display_cols].sort_values("Order_Date"), hide_index=True, width="stretch", height=340)
    st.download_button(
        "Download filtered anonymized CSV",
        filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="quick_commerce_filtered_orders.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Source: supplied Retail and wherehouse Sale_DS10.csv. Static project exhibit; "
    "no live order feed or automatic retraining. Numerical model results are fixed holdout evaluations."
)
