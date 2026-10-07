import sqlite3
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.06);
}

.metric-title {
    font-size: 14px;
    color: #6b7280;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 5px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE CONNECTION
# ============================================================

DB_PATH = r"C:\Users\pc\Downloads\customer_churn.db.db"

conn = sqlite3.connect(DB_PATH)


# ============================================================
# LOAD DATA
# ============================================================

df_customer = pd.read_sql(
    "SELECT * FROM db_customer",
    conn
)

df_subscription = pd.read_sql(
    "SELECT * FROM db_subscription",
    conn
)

df_support = pd.read_sql(
    "SELECT * FROM db_support",
    conn
)


# ============================================================
# DATA CLEANING
# ============================================================

# Rename customer name
if "name" in df_customer.columns:
    df_customer.rename(
        columns={"name": "customer_name"},
        inplace=True
    )


# Drop last two support columns if they exist
if len(df_support.columns) >= 2:
    df_support.drop(
        df_support.columns[-2:],
        axis=1,
        inplace=True
    )


# Convert dates
date_cols = [
    "subscription_start_date",
    "renewal_date",
    "cancellation_date"
]

for col in date_cols:
    if col in df_subscription.columns:
        df_subscription[col] = pd.to_datetime(
            df_subscription[col],
            errors="coerce"
        )


if "complaint_date" in df_support.columns:
    df_support["complaint_date"] = pd.to_datetime(
        df_support["complaint_date"],
        errors="coerce"
    )


# ============================================================
# CHURN FLAG
# ============================================================

df_subscription["churn_flag"] = np.where(
    df_subscription["cancellation_date"].notna(),
    1,
    0
)


# ============================================================
# SUPPORT DATA
# ============================================================

if "escalations" in df_support.columns:

    df_support["escalations"] = (
        df_support["escalations"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_support["complaint_count"] = (
        df_support
        .groupby("customerid")["customerid"]
        .transform("count")
    )

    df_support_agg = (
        df_support
        .sort_values("complaint_date")
        .drop_duplicates(
            subset=["customerid"],
            keep="last"
        )
        [
            [
                "customerid",
                "complaint_date",
                "escalations",
                "complaint_count"
            ]
        ]
        .rename(
            columns={
                "complaint_date": "last_complaint_date"
            }
        )
    )

else:

    df_support_agg = pd.DataFrame(
        columns=[
            "customerid",
            "last_complaint_date",
            "escalations",
            "complaint_count"
        ]
    )


# ============================================================
# MERGE DATA
# ============================================================

df = (
    df_subscription
    .merge(
        df_customer,
        on="customerid",
        how="left"
    )
    .merge(
        df_support_agg,
        on="customerid",
        how="left"
    )
)


# ============================================================
# COUNTRY MISSING VALUE
# ============================================================

if "state" in df.columns and "country" in df.columns:

    state_country_mapping = (
        df.dropna(subset=["country"])
        .set_index("state")["country"]
        .to_dict()
    )

    df["country"] = df["country"].fillna(
        df["state"].map(state_country_mapping)
    )


# ============================================================
# CHURN RISK
# ============================================================

if "churn_score" in df.columns:

    conditions = [
        df["churn_score"] >= 80,
        df["churn_score"] >= 50
    ]

    choices = [
        "High",
        "Medium"
    ]

    df["churn_risk"] = np.select(
        conditions,
        choices,
        default="Low"
    )

else:

    df["churn_risk"] = "Unknown"


# ============================================================
# TENURE
# ============================================================

today = pd.Timestamp.today()

df["tenure_days"] = np.where(

    df["cancellation_date"].notna(),

    (
        df["cancellation_date"]
        - df["subscription_start_date"]
    ).dt.days,

    (
        today
        - df["subscription_start_date"]
    ).dt.days
)


df["tenure_months"] = (
    df["tenure_days"] / 30
).round(1)


# ============================================================
# CLOSE CONNECTION
# ============================================================

conn.close()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎯 Churn Dashboard")

st.sidebar.markdown(
    "### Filters"
)


# Plan filter
if "plan_type" in df.columns:

    plans = sorted(
        df["plan_type"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_plans = st.sidebar.multiselect(
        "Select Plan",
        plans,
        default=plans
    )

else:

    selected_plans = []


# Contract filter
if "contract_type" in df.columns:

    contracts = sorted(
        df["contract_type"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_contracts = st.sidebar.multiselect(
        "Select Contract",
        contracts,
        default=contracts
    )

else:

    selected_contracts = []


# Gender filter
if "gender" in df.columns:

    genders = sorted(
        df["gender"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_gender = st.sidebar.multiselect(
        "Select Gender",
        genders,
        default=genders
    )

else:

    selected_gender = []


# Churn risk
risk_options = [
    "Low",
    "Medium",
    "High"
]

selected_risk = st.sidebar.multiselect(
    "Churn Risk",
    risk_options,
    default=risk_options
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_plans and "plan_type" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["plan_type"].isin(selected_plans)
    ]


if selected_contracts and "contract_type" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["contract_type"].isin(selected_contracts)
    ]


if selected_gender and "gender" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["gender"].isin(selected_gender)
    ]


if selected_risk:

    filtered_df = filtered_df[
        filtered_df["churn_risk"].isin(selected_risk)
    ]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">📊 Customer Churn Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Monitor customer churn, revenue risk, customer behavior and retention.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)

if total_customers > 0:

    churn_rate = (
        filtered_df["churn_flag"].mean() * 100
    )

    retention_rate = 100 - churn_rate

else:

    churn_rate = 0
    retention_rate = 0


if "monthly_charges" in filtered_df.columns:

    arpu = filtered_df["monthly_charges"].mean()

else:

    arpu = 0


if "monthly_charges" in filtered_df.columns:

    revenue_at_risk = filtered_df.loc[
        filtered_df["churn_flag"] == 1,
        "monthly_charges"
    ].sum()

else:

    revenue_at_risk = 0


if "tenure_months" in filtered_df.columns:

    avg_tenure = filtered_df["tenure_months"].mean()

else:

    avg_tenure = 0


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">👥 Customers</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">📉 Churn Rate</div>
            <div class="metric-value">{churn_rate:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">🔄 Retention Rate</div>
            <div class="metric-value">{retention_rate:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">💰 ARPU</div>
            <div class="metric-value">₹{arpu:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col5:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">⚠️ Revenue at Risk</div>
            <div class="metric-value">₹{revenue_at_risk:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")


# ============================================================
# ROW 1
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# CHURN TREND
# ------------------------------------------------------------

with col1:

    st.markdown(
        '<div class="section-title">📈 Monthly Churn Trend</div>',
        unsafe_allow_html=True
    )

    if "cancellation_date" in filtered_df.columns:

        trend_df = filtered_df[
            filtered_df["churn_flag"] == 1
        ].copy()

        trend_df["cancellation_month"] = (
            trend_df["cancellation_date"]
            .dt.to_period("M")
            .astype(str)
        )

        churn_trend = (
            trend_df
            .groupby("cancellation_month")
            .size()
            .reset_index(name="churned_customers")
        )

        if not churn_trend.empty:

            fig = px.line(
                churn_trend,
                x="cancellation_month",
                y="churned_customers",
                markers=True,
                title=""
            )

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Churned Customers",
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No churn data available.")


# ------------------------------------------------------------
# CHURN BY PLAN
# ------------------------------------------------------------

with col2:

    st.markdown(
        '<div class="section-title">📊 Churn Rate by Plan</div>',
        unsafe_allow_html=True
    )

    if "plan_type" in filtered_df.columns:

        churn_plan = (
            filtered_df
            .groupby("plan_type")["churn_flag"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
        )

        churn_plan.columns = [
            "plan_type",
            "churn_rate"
        ]

        fig = px.bar(
            churn_plan,
            x="plan_type",
            y="churn_rate",
            text="churn_rate",
            title=""
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Plan",
            yaxis_title="Churn Rate (%)",
            height=380
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ROW 2
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# CHURN RISK
# ------------------------------------------------------------

with col1:

    st.markdown(
        '<div class="section-title">🚨 Customer Churn Risk</div>',
        unsafe_allow_html=True
    )

    risk_df = (
        filtered_df["churn_risk"]
        .value_counts()
        .reset_index()
    )

    risk_df.columns = [
        "risk",
        "customers"
    ]

    fig = px.pie(
        risk_df,
        names="risk",
        values="customers",
        hole=0.55
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# CONTRACT TYPE
# ------------------------------------------------------------

with col2:

    st.markdown(
        '<div class="section-title">📋 Churn by Contract Type</div>',
        unsafe_allow_html=True
    )

    if "contract_type" in filtered_df.columns:

        contract_df = (
            filtered_df
            .groupby("contract_type")["churn_flag"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
        )

        contract_df.columns = [
            "contract_type",
            "churn_rate"
        ]

        fig = px.bar(
            contract_df,
            x="contract_type",
            y="churn_rate",
            text="churn_rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Contract Type",
            yaxis_title="Churn Rate (%)",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ROW 3
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# MONTHLY CHARGES vs CHURN
# ------------------------------------------------------------

with col1:

    st.markdown(
        '<div class="section-title">💰 Monthly Charges vs Churn</div>',
        unsafe_allow_html=True
    )

    if "monthly_charges" in filtered_df.columns:

        fig = px.box(
            filtered_df,
            x="churn_flag",
            y="monthly_charges",
            points="outliers"
        )

        fig.update_layout(
            xaxis_title="Churn Status",
            yaxis_title="Monthly Charges",
            height=400
        )

        fig.update_xaxes(
            tickvals=[0, 1],
            ticktext=["Stayed", "Churned"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ------------------------------------------------------------
# TENURE vs CHURN
# ------------------------------------------------------------

with col2:

    st.markdown(
        '<div class="section-title">⏳ Tenure vs Churn</div>',
        unsafe_allow_html=True
    )

    fig = px.box(
        filtered_df,
        x="churn_flag",
        y="tenure_months",
        points="outliers"
    )

    fig.update_layout(
        xaxis_title="Churn Status",
        yaxis_title="Tenure (Months)",
        height=400
    )

    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=["Stayed", "Churned"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

st.markdown(
    '<div class="section-title">🔥 Correlation Analysis</div>',
    unsafe_allow_html=True
)


numeric_df = filtered_df.select_dtypes(
    include=np.number
).copy()


if not numeric_df.empty:

    correlation = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            text=np.round(
                correlation.values,
                2
            ),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1
        )
    )

    fig.update_layout(
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ESCALATION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">📞 Support Escalation Analysis</div>',
    unsafe_allow_html=True
)


if "escalations" in filtered_df.columns:

    escalation_df = filtered_df.copy()

    escalation_df["escalation_flag"] = np.where(
        escalation_df["escalations"] == "Y",
        "Escalated",
        "Not Escalated"
    )

    escalation_summary = (
        escalation_df
        .groupby("escalation_flag")["churn_flag"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
    )

    escalation_summary.columns = [
        "escalation_status",
        "churn_rate"
    ]

    fig = px.bar(
        escalation_summary,
        x="escalation_status",
        y="churn_rate",
        text="churn_rate"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Support Status",
        yaxis_title="Churn Rate (%)",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CUSTOMER DATA
# ============================================================

st.markdown(
    '<div class="section-title">👥 Customer Details</div>',
    unsafe_allow_html=True
)


display_columns = [
    "customerid",
    "customer_name",
    "gender",
    "country",
    "state",
    "plan_type",
    "contract_type",
    "monthly_charges",
    "churn_score",
    "churn_risk",
    "churn_flag",
    "tenure_months"
]


display_columns = [
    col for col in display_columns
    if col in filtered_df.columns
]


st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    height=450
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Export Data</div>',
    unsafe_allow_html=True
)


csv = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download Filtered Customer Data",
    data=csv,
    file_name="customer_churn_filtered.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <p style="color:#6b7280;">
    Customer Churn Analytics Dashboard | Built with Python, Pandas,
    SQLite, Streamlit & Plotly
    </p>
    </center>
    """,
    unsafe_allow_html=True
)