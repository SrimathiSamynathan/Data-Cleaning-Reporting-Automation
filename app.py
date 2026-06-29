import io
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Thiranex | Data Cleaning & Reporting Automation",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — Thiranex dark branding
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0B0E1A;
    }
    .main-header {
        background: linear-gradient(135deg, #0F1530 0%, #161B33 100%);
        padding: 28px 32px;
        border-radius: 14px;
        border: 1px solid #2A3158;
        margin-bottom: 24px;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #3B82F6, #60A5FA, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #9CA3AF;
        margin-top: 6px;
        font-size: 15px;
    }
    .metric-card {
        background: #161B33;
        border: 1px solid #2A3158;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card .label {
        color: #9CA3AF;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        color: #E5E7EB;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-card .delta-good {
        color: #34D399;
        font-size: 13px;
    }
    .metric-card .delta-bad {
        color: #F87171;
        font-size: 13px;
    }
    .section-title {
        color: #93C5FD;
        font-size: 20px;
        font-weight: 700;
        margin-top: 28px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
        padding-left: 12px;
    }
    .log-box {
        background: #0F1530;
        border: 1px solid #2A3158;
        border-radius: 10px;
        padding: 14px 18px;
        color: #C7D2FE;
        font-size: 14px;
        font-family: 'Courier New', monospace;
    }
    .footer-note {
        text-align: center;
        color: #6B7280;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #2A3158;
    }
    div[data-testid="stMetricValue"] {
        color: #E5E7EB;
    }
    .stButton button {
        background: linear-gradient(90deg, #3B82F6, #2563EB);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
    }
    .stDownloadButton button {
        background: #161B33;
        color: #93C5FD;
        border: 1px solid #3B82F6;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🧹 Data Cleaning & Reporting Automation</h1>
    <p>Thiranex Data Science Internship — Task 4 · Automated preprocessing, validation & visual reporting</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SAMPLE MESSY SALES DATA GENERATOR
# ----------------------------------------------------------------------------
def generate_messy_sales_data(n=400, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    regions = ["North", "South", "East", "West"]
    region_variants = lambda r: random.choice([r, r.upper(), r.lower(), f" {r}", f"{r} "])

    categories = ["Electronics", "Apparel", "Home & Kitchen", "Sports", "Beauty"]
    products_by_cat = {
        "Electronics": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Power Bank"],
        "Apparel": ["Cotton T-Shirt", "Denim Jacket", "Running Shoes", "Hoodie"],
        "Home & Kitchen": ["Air Fryer", "Coffee Maker", "Blender", "Cookware Set"],
        "Sports": ["Yoga Mat", "Dumbbell Set", "Cycling Helmet", "Football"],
        "Beauty": ["Face Serum", "Hair Dryer", "Lipstick Set", "Sunscreen"],
    }

    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"]
    start_date = datetime(2025, 1, 1)

    rows = []
    for i in range(n):
        region = random.choice(regions)
        category = random.choice(categories)
        product = random.choice(products_by_cat[category])
        qty = random.randint(1, 12)
        unit_price = round(random.uniform(8, 250), 2)

        # inconsistent currency formatting
        if random.random() < 0.3:
            price_str = f"${unit_price:,.2f}"
        else:
            price_str = str(unit_price)

        # inconsistent date formatting
        d = start_date + timedelta(days=random.randint(0, 364))
        fmt = random.choice(date_formats)
        date_str = d.strftime(fmt)

        revenue = round(qty * unit_price, 2)

        row = {
            "OrderID": f"ORD{1000 + i}",
            "OrderDate": date_str,
            "Region": region_variants(region),
            "Category": category,
            "Product": f" {product} " if random.random() < 0.15 else product,
            "Quantity": qty,
            "UnitPrice": price_str,
            "Revenue": revenue,
            "CustomerName": f"Customer{i%180}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # inject missing values
    for col in ["Region", "UnitPrice", "Quantity", "Revenue", "CustomerName"]:
        idx = df.sample(frac=0.06, random_state=seed).index
        df.loc[idx, col] = np.nan

    # inject exact duplicate rows
    dup_rows = df.sample(frac=0.05, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------------
# CLEANING PIPELINE
# ----------------------------------------------------------------------------
def clean_currency(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def parse_mixed_dates(series):
    return pd.to_datetime(series, errors="coerce", format="mixed")


def run_cleaning_pipeline(df_raw, fill_strategy="median"):
    log = []
    df = df_raw.copy()
    rows_before = len(df)

    # 1. Strip whitespace from string/object columns
    str_cols = df.select_dtypes(include="object").columns
    whitespace_fixed = 0
    for col in str_cols:
        before = df[col].astype(str)
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None"]), col] = np.nan
        whitespace_fixed += (before.str.strip() != before).sum()
    log.append(f"Trimmed leading/trailing whitespace across {len(str_cols)} text columns ({whitespace_fixed} values affected).")

    # 2. Standardize categorical casing (Region, Category)
    case_fixed = 0
    for col in ["Region", "Category"]:
        if col in df.columns:
            before = df[col].copy()
            df[col] = df[col].str.title()
            case_fixed += (before != df[col]).sum()
    log.append(f"Standardized text casing for categorical fields ({case_fixed} values normalized).")

    # 3. Parse inconsistent date formats
    if "OrderDate" in df.columns:
        parsed = parse_mixed_dates(df["OrderDate"])
        unparsed = parsed.isna().sum() - df["OrderDate"].isna().sum()
        df["OrderDate"] = parsed.dt.strftime("%Y-%m-%d")
        log.append(f"Normalized OrderDate to ISO format (YYYY-MM-DD); {unparsed} unparsable dates flagged.")

    # 4. Clean currency / numeric fields
    if "UnitPrice" in df.columns:
        before_missing = df["UnitPrice"].isna().sum()
        df["UnitPrice"] = df["UnitPrice"].apply(clean_currency)
        log.append("Converted UnitPrice to numeric float, removing currency symbols and thousand separators.")

    for col in ["Quantity", "Revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Remove exact duplicate rows
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"Removed {dup_count} exact duplicate rows.")

    # 6. Recompute Revenue where missing but Quantity & UnitPrice available
    if {"Quantity", "UnitPrice", "Revenue"}.issubset(df.columns):
        mask = df["Revenue"].isna() & df["Quantity"].notna() & df["UnitPrice"].notna()
        recomputed = mask.sum()
        df.loc[mask, "Revenue"] = (df.loc[mask, "Quantity"] * df.loc[mask, "UnitPrice"]).round(2)
        log.append(f"Recomputed {recomputed} missing Revenue values from Quantity x UnitPrice.")

    # 7. Handle remaining missing values
    missing_handled = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            fill_val = df[col].median() if fill_strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(round(fill_val, 2))
            missing_handled[col] = n_missing

    categorical_cols = [c for c in ["Region", "CustomerName", "Category", "Product"] if c in df.columns]
    for col in categorical_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            missing_handled[col] = n_missing

    for col, n in missing_handled.items():
        strategy_note = "median/mean" if col in numeric_cols else "most frequent value"
        log.append(f"Filled {n} missing value(s) in '{col}' using {strategy_note}.")

    rows_after = len(df)
    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": int(dup_count),
        "missing_values_fixed": int(sum(missing_handled.values())),
        "columns_affected": len(missing_handled),
    }
    return df, log, summary


# ----------------------------------------------------------------------------
# SIDEBAR — DATA SOURCE
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 📂 Data Source")
data_source = st.sidebar.radio(
    "Choose input data",
    ["Use sample messy sales data", "Upload my own CSV"],
    index=0,
)

uploaded_file = None
if data_source == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.sidebar.markdown("### ⚙️ Cleaning Settings")
fill_strategy = st.sidebar.selectbox("Missing numeric value strategy", ["median", "mean"], index=0)
run_clicked = st.sidebar.button("🚀 Run Automated Cleaning", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Thiranex Internship · Task 4 of N\nSkill Development & Future Tech")

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = generate_messy_sales_data()

if data_source == "Upload my own CSV" and uploaded_file is not None:
    st.session_state.raw_df = pd.read_csv(uploaded_file)
elif data_source == "Use sample messy sales data" and "cleaned_df" not in st.session_state:
    st.session_state.raw_df = generate_messy_sales_data()

raw_df = st.session_state.raw_df

# ----------------------------------------------------------------------------
# RAW DATA QUALITY OVERVIEW
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">📋 Raw Data Overview</div>', unsafe_allow_html=True)

missing_per_col = raw_df.isna().sum()
total_missing = int(missing_per_col.sum())
duplicate_rows = int(raw_df.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="label">Total Rows</div><div class="value">{len(raw_df)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="label">Total Columns</div><div class="value">{raw_df.shape[1]}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="label">Missing Values</div><div class="value">{total_missing}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="label">Duplicate Rows</div><div class="value">{duplicate_rows}</div></div>', unsafe_allow_html=True)

with st.expander("Preview raw (uncleaned) data", expanded=False):
    st.dataframe(raw_df.head(20), use_container_width=True)

if total_missing > 0:
    missing_chart_df = missing_per_col[missing_per_col > 0].sort_values(ascending=True)
    fig_missing = go.Figure(go.Bar(
        x=missing_chart_df.values,
        y=missing_chart_df.index,
        orientation="h",
        marker_color="#F87171",
    ))
    fig_missing.update_layout(
        title="Missing Values by Column (Before Cleaning)",
        template="plotly_dark",
        paper_bgcolor="#0B0E1A",
        plot_bgcolor="#0B0E1A",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_missing, use_container_width=True)

# ----------------------------------------------------------------------------
# RUN CLEANING
# ----------------------------------------------------------------------------
if run_clicked or "cleaned_df" not in st.session_state:
    cleaned_df, log, summary = run_cleaning_pipeline(raw_df, fill_strategy)
    st.session_state.cleaned_df = cleaned_df
    st.session_state.log = log
    st.session_state.summary = summary

cleaned_df = st.session_state.cleaned_df
log = st.session_state.log
summary = st.session_state.summary

# ----------------------------------------------------------------------------
# BEFORE / AFTER COMPARISON
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">✅ Cleaning Results</div>', unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)
with b1:
    st.markdown(f'<div class="metric-card"><div class="label">Rows Before</div><div class="value">{summary["rows_before"]}</div></div>', unsafe_allow_html=True)
with b2:
    st.markdown(f'<div class="metric-card"><div class="label">Rows After</div><div class="value">{summary["rows_after"]}</div><div class="delta-good">−{summary["duplicates_removed"]} duplicates</div></div>', unsafe_allow_html=True)
with b3:
    st.markdown(f'<div class="metric-card"><div class="label">Missing Values Fixed</div><div class="value">{summary["missing_values_fixed"]}</div><div class="delta-good">0 remaining</div></div>', unsafe_allow_html=True)
with b4:
    st.markdown(f'<div class="metric-card"><div class="label">Columns Cleaned</div><div class="value">{summary["columns_affected"]}</div></div>', unsafe_allow_html=True)

st.markdown("##### 🧾 Cleaning Log")
log_html = "<br>".join([f"→ {item}" for item in log])
st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

with st.expander("Preview cleaned data", expanded=True):
    st.dataframe(cleaned_df.head(20), use_container_width=True)

# ----------------------------------------------------------------------------
# VISUAL REPORTING (on cleaned data)
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">📊 Automated Visual Report</div>', unsafe_allow_html=True)

has_region = "Region" in cleaned_df.columns
has_revenue = "Revenue" in cleaned_df.columns
has_category = "Category" in cleaned_df.columns
has_date = "OrderDate" in cleaned_df.columns

v1, v2 = st.columns(2)

if has_region and has_revenue:
    region_rev = cleaned_df.groupby("Region")["Revenue"].sum().sort_values(ascending=False).reset_index()
    fig_region = px.bar(
        region_rev, x="Region", y="Revenue",
        title="Revenue by Region",
        color="Region",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_region.update_layout(template="plotly_dark", paper_bgcolor="#0B0E1A", plot_bgcolor="#0B0E1A", height=350)
    v1.plotly_chart(fig_region, use_container_width=True)

if has_category and has_revenue:
    cat_rev = cleaned_df.groupby("Category")["Revenue"].sum().reset_index()
    fig_cat = px.pie(
        cat_rev, names="Category", values="Revenue",
        title="Revenue Share by Category",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_cat.update_layout(template="plotly_dark", paper_bgcolor="#0B0E1A", plot_bgcolor="#0B0E1A", height=350)
    v2.plotly_chart(fig_cat, use_container_width=True)

if has_date and has_revenue:
    trend_df = cleaned_df.copy()
    trend_df["OrderDate"] = pd.to_datetime(trend_df["OrderDate"], errors="coerce")
    trend_df = trend_df.dropna(subset=["OrderDate"])
    monthly = trend_df.set_index("OrderDate").resample("ME")["Revenue"].sum().reset_index()
    fig_trend = px.line(
        monthly, x="OrderDate", y="Revenue",
        title="Monthly Revenue Trend (Post-Cleaning)",
        markers=True,
    )
    fig_trend.update_traces(line_color="#3B82F6")
    fig_trend.update_layout(template="plotly_dark", paper_bgcolor="#0B0E1A", plot_bgcolor="#0B0E1A", height=350)
    st.plotly_chart(fig_trend, use_container_width=True)

# ----------------------------------------------------------------------------
# KEY INSIGHTS
# ----------------------------------------------------------------------------
insights = []
if has_revenue:
    insights.append(f"Total revenue after cleaning: ₹{cleaned_df['Revenue'].sum():,.2f}")
if has_region and has_revenue:
    top_region = cleaned_df.groupby("Region")["Revenue"].sum().idxmax()
    insights.append(f"Top performing region: {top_region}")
if has_category and has_revenue:
    top_category = cleaned_df.groupby("Category")["Revenue"].sum().idxmax()
    insights.append(f"Best-selling category: {top_category}")

if insights:
    st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f"- {insight}")

# ----------------------------------------------------------------------------
# DOWNLOADS
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">⬇️ Export</div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)

csv_buffer = io.StringIO()
cleaned_df.to_csv(csv_buffer, index=False)
d1.download_button(
    "Download Cleaned Dataset (CSV)",
    data=csv_buffer.getvalue(),
    file_name="cleaned_sales_data.csv",
    mime="text/csv",
    use_container_width=True,
)

report_lines = [
    "THIRANEX — DATA CLEANING & REPORTING AUTOMATION",
    "Automated Data Quality Report",
    "=" * 50,
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "SUMMARY",
    "-" * 50,
    f"Rows before cleaning : {summary['rows_before']}",
    f"Rows after cleaning  : {summary['rows_after']}",
    f"Duplicates removed   : {summary['duplicates_removed']}",
    f"Missing values fixed : {summary['missing_values_fixed']}",
    f"Columns cleaned      : {summary['columns_affected']}",
    "",
    "CLEANING STEPS PERFORMED",
    "-" * 50,
] + [f"- {item}" for item in log] + [
    "",
    "KEY INSIGHTS",
    "-" * 50,
] + [f"- {item}" for item in insights]

report_text = "\n".join(report_lines)
d2.download_button(
    "Download Automated Report (.txt)",
    data=report_text,
    file_name="data_cleaning_report.txt",
    mime="text/plain",
    use_container_width=True,
)

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="footer-note">
    Thiranex — Skill Development & Future Tech<br>
    Data Science Internship · Task 4: Data Cleaning & Reporting Automation
</div>
""", unsafe_allow_html=True)
