# =============================================================================
#  CareSight AI — Interactive Healthcare Analytics & Patient Readmission
#  Intelligence Platform
#  Author  : Anjum
#  Project : IBM SkillsBuild Data Analytics with AI Internship
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import io

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareSight AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2342 0%, #1a3a5c 60%, #1e5799 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 15px; }

    /* KPI card */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 18px 16px 18px;
        border-left: 5px solid #1e5799;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 6px;
    }
    .kpi-title { font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #0a2342; margin: 4px 0 0 0; }
    .kpi-sub   { font-size: 12px; color: #9ca3af; margin-top: 2px; }

    /* Section header */
    .section-header {
        background: linear-gradient(90deg, #0a2342, #1e5799);
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 700;
        margin: 18px 0 12px 0;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #fff8e1;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #78350f;
        margin-top: 16px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 12px;
        color: #9ca3af;
        padding: 20px 0 8px 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 30px;
    }

    /* Chart card */
    .chart-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
CSV_FILENAME = "Hospital Patient .csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        st.error(
            f"❌ Dataset file **'{CSV_FILENAME}'** not found.\n\n"
            "Please place the CSV file in the **same folder** as this script and restart."
        )
        st.stop()

    # Strip column name whitespace
    df.columns = df.columns.str.strip()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Parse dates
    df["Admission_Date"]  = pd.to_datetime(df["Admission_Date"],  format="%d-%m-%Y", errors="coerce")
    df["Discharge_Date"]  = pd.to_datetime(df["Discharge_Date"],  format="%d-%m-%Y", errors="coerce")

    # Ensure numeric types
    df["Age"]             = pd.to_numeric(df["Age"], errors="coerce")
    df["Length_of_Stay"]  = pd.to_numeric(df["Length_of_Stay"], errors="coerce")
    df["Satisfaction"]    = pd.to_numeric(df["Satisfaction"], errors="coerce")
    df["Total_Cost"]      = pd.to_numeric(df["Total_Cost"], errors="coerce")
    df["Year_of_Admission"] = pd.to_numeric(df["Year_of_Admission"], errors="coerce")

    # Binary target
    df["Readmission_Binary"] = df["Readmission"].map({"Yes": 1, "No": 0})

    # Age group
    bins   = [0, 18, 35, 50, 65, 120]
    labels = ["<18", "18-35", "36-50", "51-65", "65+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    return df


df = load_data()

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = [
    "#1e5799", "#2980b9", "#27ae60", "#e74c3c",
    "#f39c12", "#8e44ad", "#16a085", "#d35400",
    "#2c3e50", "#c0392b", "#1abc9c", "#e67e22"
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 6px 0;'>
        <div style='font-size:38px;'>🏥</div>
        <div style='font-size:22px; font-weight:800; color:#e8f0fe;'>CareSight AI</div>
        <div style='font-size:11px; color:#93c5fd; margin-top:2px;'>Healthcare Intelligence, Simplified.</div>
    </div>
    <hr style='border-color:#2d6ab4; margin: 10px 0 16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Patient Analytics", "🤖 AI Insights",
         "📈 Model Performance", "ℹ️ About Project"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2d6ab4; margin: 16px 0 10px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:12px; color:#93c5fd;'>
        📋 <b>{len(df):,}</b> Patient Records<br>
        📅 Years: <b>{int(df['Year_of_Admission'].min())} – {int(df['Year_of_Admission'].max())}</b><br>
        🏷️ <b>{df['Condition'].nunique()}</b> Medical Conditions<br>
        🗺️ <b>{df['Patient_State'].nunique()}</b> States
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:11px; color:#93c5fd; margin-top:14px; font-style:italic;'>
        ⚠️ Educational use only.<br>Not clinical advice.
    </div>
    """, unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def kpi_card(title, value, sub=""):
    return f"""
    <div class='kpi-card'>
        <div class='kpi-title'>{title}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>"""

def section_header(text):
    st.markdown(f"<div class='section-header'>{text}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("""
    <h1 style='color:#0a2342; margin-bottom:2px;'>📊 CareSight AI — Dashboard</h1>
    <p style='color:#6b7280; font-size:15px; margin-top:0;'>
        Real-time overview of hospital patient analytics derived from <b>984 patient records</b>.
    </p>
    """, unsafe_allow_html=True)

    # KPI row
    total_patients    = len(df)
    readmission_rate  = df["Readmission_Binary"].mean() * 100
    avg_los           = df["Length_of_Stay"].mean()
    avg_cost          = df["Total_Cost"].mean()
    avg_satisfaction  = df["Satisfaction"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Total Patients", f"{total_patients:,}", "All records"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Readmission Rate", f"{readmission_rate:.1f}%", "Of all admissions"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg Length of Stay", f"{avg_los:.1f} days", "Per admission"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Avg Healthcare Cost", f"₹{avg_cost:,.0f}", "Per patient"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Avg Satisfaction", f"{avg_satisfaction:.2f} / 5", "Patient rating"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 — Condition distribution & Readmission breakdown
    section_header("📌 Patient Distribution")
    col1, col2 = st.columns(2)

    with col1:
        cond_counts = df["Condition"].value_counts().reset_index()
        cond_counts.columns = ["Condition", "Count"]
        fig = px.bar(
            cond_counts, x="Count", y="Condition", orientation="h",
            title="Patients by Medical Condition",
            color="Count", color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig.update_layout(showlegend=False, height=380, coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        readmit_counts = df["Readmission"].value_counts().reset_index()
        readmit_counts.columns = ["Readmission", "Count"]
        fig2 = px.pie(
            readmit_counts, names="Readmission", values="Count",
            title="Readmission Distribution",
            color_discrete_sequence=["#1e5799", "#e74c3c"],
            template="plotly_white", hole=0.4
        )
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2 — Outcome & Cost by condition
    col3, col4 = st.columns(2)

    with col3:
        outcome_counts = df["Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig3 = px.pie(
            outcome_counts, names="Outcome", values="Count",
            title="Patient Outcomes",
            color_discrete_sequence=["#27ae60", "#f39c12", "#e74c3c"],
            template="plotly_white", hole=0.4
        )
        fig3.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        cost_cond = df.groupby("Condition")["Total_Cost"].mean().reset_index()
        cost_cond.columns = ["Condition", "Avg_Cost"]
        cost_cond = cost_cond.sort_values("Avg_Cost", ascending=True)
        fig4 = px.bar(
            cost_cond, x="Avg_Cost", y="Condition", orientation="h",
            title="Average Cost by Medical Condition (₹)",
            color="Avg_Cost", color_continuous_scale="Oranges",
            template="plotly_white"
        )
        fig4.update_layout(showlegend=False, height=340, coloraxis_showscale=False,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3 — Length of stay & Satisfaction
    col5, col6 = st.columns(2)

    with col5:
        los_cond = df.groupby("Condition")["Length_of_Stay"].mean().reset_index()
        los_cond.columns = ["Condition", "Avg_LOS"]
        los_cond = los_cond.sort_values("Avg_LOS", ascending=False)
        fig5 = px.bar(
            los_cond, x="Condition", y="Avg_LOS",
            title="Average Length of Stay by Condition (days)",
            color="Avg_LOS", color_continuous_scale="Teal",
            template="plotly_white"
        )
        fig5.update_layout(showlegend=False, height=340, coloraxis_showscale=False,
                           xaxis_tickangle=-35, margin=dict(l=10, r=10, t=40, b=60))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        sat_cond = df.groupby("Condition")["Satisfaction"].mean().reset_index()
        sat_cond.columns = ["Condition", "Avg_Satisfaction"]
        sat_cond = sat_cond.sort_values("Avg_Satisfaction", ascending=False)
        fig6 = px.bar(
            sat_cond, x="Condition", y="Avg_Satisfaction",
            title="Average Satisfaction Score by Condition (out of 5)",
            color="Avg_Satisfaction",
            color_continuous_scale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1, "#27ae60"]],
            template="plotly_white"
        )
        fig6.update_layout(showlegend=False, height=340, coloraxis_showscale=False,
                           xaxis_tickangle=-35, margin=dict(l=10, r=10, t=40, b=60))
        st.plotly_chart(fig6, use_container_width=True)

    # Yearly trend
    section_header("📅 Yearly Admission Trend")
    yearly = df.groupby("Year_of_Admission").size().reset_index(name="Admissions")
    fig7 = px.line(
        yearly, x="Year_of_Admission", y="Admissions",
        title="Total Admissions per Year", markers=True,
        color_discrete_sequence=["#1e5799"], template="plotly_white"
    )
    fig7.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    fig7.update_xaxes(type="category")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("<div class='footer'>CareSight AI — IBM SkillsBuild Data Analytics Internship Project | Educational Use Only</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Patient Analytics":
    st.markdown("""
    <h1 style='color:#0a2342; margin-bottom:2px;'>🔍 Patient Analytics</h1>
    <p style='color:#6b7280; font-size:15px; margin-top:0;'>
        Filter and explore patient data interactively. All charts update in real-time.
    </p>
    """, unsafe_allow_html=True)

    # ── Filters ──
    section_header("🎛️ Filter Panel")
    f1, f2, f3, f4 = st.columns(4)
    f5, f6, f7     = st.columns(3)

    with f1:
        years = ["All"] + sorted(df["Year_of_Admission"].dropna().astype(int).unique().tolist())
        sel_year = st.selectbox("Year of Admission", years)
    with f2:
        genders = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
        sel_gender = st.selectbox("Gender", genders)
    with f3:
        conditions = ["All"] + sorted(df["Condition"].dropna().unique().tolist())
        sel_cond = st.selectbox("Medical Condition", conditions)
    with f4:
        states = ["All"] + sorted(df["Patient_State"].dropna().unique().tolist())
        sel_state = st.selectbox("Patient State", states)
    with f5:
        readmit_opts = ["All", "Yes", "No"]
        sel_readmit = st.selectbox("Readmission", readmit_opts)
    with f6:
        outcomes = ["All"] + sorted(df["Outcome"].dropna().unique().tolist())
        sel_outcome = st.selectbox("Outcome", outcomes)
    with f7:
        insured_opts = ["All", "Yes", "No"]
        sel_insured = st.selectbox("Insurance Claimed", insured_opts)

    # Apply filters
    fdf = df.copy()
    if sel_year      != "All": fdf = fdf[fdf["Year_of_Admission"] == int(sel_year)]
    if sel_gender    != "All": fdf = fdf[fdf["Gender"]            == sel_gender]
    if sel_cond      != "All": fdf = fdf[fdf["Condition"]         == sel_cond]
    if sel_state     != "All": fdf = fdf[fdf["Patient_State"]     == sel_state]
    if sel_readmit   != "All": fdf = fdf[fdf["Readmission"]       == sel_readmit]
    if sel_outcome   != "All": fdf = fdf[fdf["Outcome"]           == sel_outcome]
    if sel_insured   != "All": fdf = fdf[fdf["Insurance_Claimed"] == sel_insured]

    # Filtered KPIs
    st.markdown(f"<p style='color:#1e5799; font-weight:600; font-size:14px;'>Showing <b>{len(fdf):,}</b> records after filtering.</p>",
                unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi_card("Filtered Patients", f"{len(fdf):,}", ""), unsafe_allow_html=True)
    with k2: st.markdown(kpi_card("Readmission Rate", f"{fdf['Readmission_Binary'].mean()*100:.1f}%", ""), unsafe_allow_html=True)
    with k3: st.markdown(kpi_card("Avg Age", f"{fdf['Age'].mean():.1f} yrs", ""), unsafe_allow_html=True)
    with k4: st.markdown(kpi_card("Avg LOS", f"{fdf['Length_of_Stay'].mean():.1f} days", ""), unsafe_allow_html=True)
    with k5: st.markdown(kpi_card("Avg Cost", f"₹{fdf['Total_Cost'].mean():,.0f}", ""), unsafe_allow_html=True)

    if len(fdf) == 0:
        st.warning("No records match the selected filters. Please adjust your filter criteria.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("👤 Demographics & Condition Analysis")
        c1, c2 = st.columns(2)

        with c1:
            # Age distribution
            fig = px.histogram(
                fdf, x="Age", nbins=20, color_discrete_sequence=["#1e5799"],
                title="Age Distribution", template="plotly_white"
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Gender split
            gen_counts = fdf["Gender"].value_counts().reset_index()
            gen_counts.columns = ["Gender", "Count"]
            fig = px.pie(
                gen_counts, names="Gender", values="Count",
                title="Gender Distribution", hole=0.4,
                color_discrete_sequence=["#1e5799", "#e91e8c"],
                template="plotly_white"
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # Readmission by condition
            r_cond = fdf.groupby("Condition")["Readmission_Binary"].mean().reset_index()
            r_cond.columns = ["Condition", "Readmission_Rate"]
            r_cond["Readmission_Rate"] = r_cond["Readmission_Rate"] * 100
            r_cond = r_cond.sort_values("Readmission_Rate", ascending=False)
            fig = px.bar(
                r_cond, x="Condition", y="Readmission_Rate",
                title="Readmission Rate by Condition (%)",
                color="Readmission_Rate",
                color_continuous_scale=[[0, "#27ae60"], [0.5, "#f39c12"], [1, "#e74c3c"]],
                template="plotly_white"
            )
            fig.update_layout(height=320, coloraxis_showscale=False,
                              xaxis_tickangle=-35, margin=dict(l=10, r=10, t=40, b=60))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            # Age group vs Readmission
            ag_r = fdf.groupby("Age_Group", observed=True)["Readmission_Binary"].mean().reset_index()
            ag_r.columns = ["Age_Group", "Readmission_Rate"]
            ag_r["Readmission_Rate"] = ag_r["Readmission_Rate"] * 100
            fig = px.bar(
                ag_r, x="Age_Group", y="Readmission_Rate",
                title="Readmission Rate by Age Group (%)",
                color="Readmission_Rate",
                color_continuous_scale=[[0, "#27ae60"], [1, "#e74c3c"]],
                template="plotly_white"
            )
            fig.update_layout(height=320, coloraxis_showscale=False,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        section_header("💰 Cost & Stay Analysis")
        c5, c6 = st.columns(2)

        with c5:
            fig = px.box(
                fdf, x="Condition", y="Total_Cost",
                title="Cost Distribution by Condition (₹)",
                color="Condition", color_discrete_sequence=PALETTE,
                template="plotly_white"
            )
            fig.update_layout(height=350, showlegend=False,
                              xaxis_tickangle=-35, margin=dict(l=10, r=10, t=40, b=60))
            st.plotly_chart(fig, use_container_width=True)

        with c6:
            fig = px.box(
                fdf, x="Condition", y="Length_of_Stay",
                title="Length of Stay Distribution by Condition",
                color="Condition", color_discrete_sequence=PALETTE,
                template="plotly_white"
            )
            fig.update_layout(height=350, showlegend=False,
                              xaxis_tickangle=-35, margin=dict(l=10, r=10, t=40, b=60))
            st.plotly_chart(fig, use_container_width=True)

        section_header("📊 Satisfaction & Insurance")
        c7, c8 = st.columns(2)

        with c7:
            sat_dist = fdf["Satisfaction"].value_counts().sort_index().reset_index()
            sat_dist.columns = ["Satisfaction Score", "Count"]
            fig = px.bar(
                sat_dist, x="Satisfaction Score", y="Count",
                title="Satisfaction Score Distribution",
                color="Count", color_continuous_scale="Viridis",
                template="plotly_white"
            )
            fig.update_layout(height=300, coloraxis_showscale=False,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c8:
            ins = fdf["Insurance_Claimed"].value_counts().reset_index()
            ins.columns = ["Insurance Claimed", "Count"]
            fig = px.pie(
                ins, names="Insurance Claimed", values="Count",
                title="Insurance Claimed Distribution", hole=0.4,
                color_discrete_sequence=["#1e5799", "#e74c3c"],
                template="plotly_white"
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Yearly admissions trend (filtered)
        section_header("📅 Yearly Admission Trend (Filtered)")
        yr_fdf = fdf.groupby("Year_of_Admission").size().reset_index(name="Admissions")
        fig = px.area(
            yr_fdf, x="Year_of_Admission", y="Admissions",
            title="Admissions Over Time (Filtered)",
            color_discrete_sequence=["#1e5799"], template="plotly_white"
        )
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='footer'>CareSight AI — IBM SkillsBuild Data Analytics Internship Project | Educational Use Only</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Insights":
    st.markdown("""
    <h1 style='color:#0a2342; margin-bottom:2px;'>🤖 AI Insights — Readmission Predictor</h1>
    <p style='color:#6b7280; font-size:15px; margin-top:0;'>
        Enter patient details to predict the likelihood of hospital readmission using trained ML models.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer-box'>
        ⚠️ <b>Disclaimer:</b> This project is intended for educational and analytical purposes.
        Predictions are based on historical data and are <b>not medical diagnosis or clinical advice</b>.
        Always consult a qualified healthcare professional for medical decisions.
    </div>
    """, unsafe_allow_html=True)

    # ── Train models (cached) ──
    @st.cache_resource
    def train_models(dataframe):
        df_ml = dataframe.copy()

        # Features — no Patient_ID, no target leakage
        feature_cols = ["Age", "Gender", "Condition", "Patient_State",
                        "Length_of_Stay", "Outcome", "Satisfaction",
                        "Insurance_Claimed", "Total_Cost", "Year_of_Admission"]
        df_ml = df_ml[feature_cols + ["Readmission_Binary"]].dropna()

        # Encode categoricals
        le_dict = {}
        cat_cols = ["Gender", "Condition", "Patient_State", "Outcome", "Insurance_Claimed"]
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le

        X = df_ml[feature_cols]
        y = df_ml["Readmission_Binary"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        return lr, rf, le_dict, feature_cols, X_train, X_test, y_train, y_test, X

    lr_model, rf_model, le_dict, feature_cols, X_train, X_test, y_train, y_test, X_full = train_models(df)

    section_header("🧾 Patient Input Form")
    st.markdown("Fill in the patient details below and click **Analyze Readmission**.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            inp_age       = st.number_input("Age", min_value=1, max_value=120, value=50)
            inp_gender    = st.selectbox("Gender", sorted(df["Gender"].dropna().unique().tolist()))
            inp_condition = st.selectbox("Medical Condition", sorted(df["Condition"].dropna().unique().tolist()))
        with col2:
            inp_state     = st.selectbox("Patient State", sorted(df["Patient_State"].dropna().unique().tolist()))
            inp_los       = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=5)
            inp_outcome   = st.selectbox("Expected Outcome", sorted(df["Outcome"].dropna().unique().tolist()))
        with col3:
            inp_sat       = st.slider("Satisfaction Score (1–5)", min_value=1, max_value=5, value=3)
            inp_insurance = st.selectbox("Insurance Claimed", ["Yes", "No"])
            inp_cost      = st.number_input("Total Cost (₹)", min_value=100, max_value=500000, value=5000, step=500)
            inp_year      = st.number_input("Year of Admission", min_value=2018, max_value=2030, value=2024)

        submitted = st.form_submit_button("🔍 Analyze Readmission", use_container_width=True)

    if submitted:
        def safe_encode(le, value):
            classes = list(le.classes_)
            if value in classes:
                return le.transform([value])[0]
            return 0  # fallback for unseen label

        row = {
            "Age":              inp_age,
            "Gender":           safe_encode(le_dict["Gender"],           inp_gender),
            "Condition":        safe_encode(le_dict["Condition"],        inp_condition),
            "Patient_State":    safe_encode(le_dict["Patient_State"],    inp_state),
            "Length_of_Stay":   inp_los,
            "Outcome":          safe_encode(le_dict["Outcome"],          inp_outcome),
            "Satisfaction":     inp_sat,
            "Insurance_Claimed": safe_encode(le_dict["Insurance_Claimed"], inp_insurance),
            "Total_Cost":       inp_cost,
            "Year_of_Admission": inp_year,
        }
        input_df = pd.DataFrame([row])

        lr_pred  = lr_model.predict(input_df)[0]
        lr_prob  = lr_model.predict_proba(input_df)[0][1]
        rf_pred  = rf_model.predict(input_df)[0]
        rf_prob  = rf_model.predict_proba(input_df)[0][1]

        section_header("📊 Prediction Results")
        res1, res2 = st.columns(2)

        for col_obj, model_name, pred, prob in [
            (res1, "Logistic Regression", lr_pred, lr_prob),
            (res2, "Random Forest",       rf_pred, rf_prob)
        ]:
            with col_obj:
                label   = "⚠️ Likely Readmission" if pred == 1 else "✅ No Readmission Expected"
                color   = "#e74c3c" if pred == 1 else "#27ae60"
                pct_str = f"{prob * 100:.1f}%"
                st.markdown(f"""
                <div style='background:white; border-radius:12px; padding:20px;
                            border-left:5px solid {color}; box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
                    <div style='font-size:13px; color:#6b7280; font-weight:600;'>{model_name}</div>
                    <div style='font-size:22px; font-weight:700; color:{color}; margin:6px 0;'>{label}</div>
                    <div style='font-size:15px; color:#374151;'>
                        Readmission Probability: <b>{pct_str}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Logistic Regression", "Random Forest"],
            y=[lr_prob * 100, rf_prob * 100],
            marker_color=["#1e5799", "#27ae60"],
            text=[f"{lr_prob*100:.1f}%", f"{rf_prob*100:.1f}%"],
            textposition="outside"
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="50% Threshold", annotation_position="right")
        fig.update_layout(
            title="Readmission Probability Comparison",
            yaxis_title="Probability (%)", yaxis_range=[0, 110],
            template="plotly_white", height=320,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='disclaimer-box'>
            ⚠️ <b>Disclaimer:</b> This project is intended for educational and analytical purposes.
            Predictions are based on historical data and are <b>not medical diagnosis or clinical advice</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='footer'>CareSight AI — IBM SkillsBuild Data Analytics Internship Project | Educational Use Only</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("""
    <h1 style='color:#0a2342; margin-bottom:2px;'>📈 Model Performance</h1>
    <p style='color:#6b7280; font-size:15px; margin-top:0;'>
        Evaluation metrics, confusion matrices, and feature importance for trained ML models.
    </p>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def get_model_results(dataframe):
        df_ml = dataframe.copy()
        feature_cols = ["Age", "Gender", "Condition", "Patient_State",
                        "Length_of_Stay", "Outcome", "Satisfaction",
                        "Insurance_Claimed", "Total_Cost", "Year_of_Admission"]
        df_ml = df_ml[feature_cols + ["Readmission_Binary"]].dropna()

        le_dict = {}
        cat_cols = ["Gender", "Condition", "Patient_State", "Outcome", "Insurance_Claimed"]
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le

        X = df_ml[feature_cols]
        y = df_ml["Readmission_Binary"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_pred  = lr.predict(X_test)
        lr_prob  = lr.predict_proba(X_test)[:, 1]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred  = rf.predict(X_test)
        rf_prob  = rf.predict_proba(X_test)[:, 1]

        return {
            "lr": lr, "rf": rf,
            "lr_pred": lr_pred, "lr_prob": lr_prob,
            "rf_pred": rf_pred, "rf_prob": rf_prob,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "X_train": X_train, "X_test": X_test
        }

    res = get_model_results(df)
    y_test   = res["y_test"]
    lr_pred  = res["lr_pred"]
    rf_pred  = res["rf_pred"]
    lr_prob  = res["lr_prob"]
    rf_prob  = res["rf_prob"]
    feature_cols = res["feature_cols"]

    # ── Metrics table ──
    section_header("📊 Model Comparison Metrics")

    def get_metrics(y_true, y_pred, y_prob):
        return {
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
        }

    lr_metrics = get_metrics(y_test, lr_pred, lr_prob)
    rf_metrics = get_metrics(y_test, rf_pred, rf_prob)

    metrics_df = pd.DataFrame({
        "Metric":              list(lr_metrics.keys()),
        "Logistic Regression": list(lr_metrics.values()),
        "Random Forest":       list(rf_metrics.values()),
    })
    st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)

    # Bar comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Logistic Regression", x=list(lr_metrics.keys()),
                         y=list(lr_metrics.values()), marker_color="#1e5799"))
    fig.add_trace(go.Bar(name="Random Forest",       x=list(rf_metrics.keys()),
                         y=list(rf_metrics.values()), marker_color="#27ae60"))
    fig.update_layout(
        barmode="group", title="Metric Comparison: LR vs RF",
        yaxis_title="Score", yaxis_range=[0, 1.05],
        template="plotly_white", height=350,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrices ──
    section_header("🔲 Confusion Matrices")
    cm1, cm2 = st.columns(2)

    for col_obj, model_name, y_pred in [
        (cm1, "Logistic Regression", lr_pred),
        (cm2, "Random Forest",       rf_pred)
    ]:
        with col_obj:
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not Readmitted", "Readmitted"],
                y=["Not Readmitted", "Readmitted"],
                color_continuous_scale="Blues",
                title=f"Confusion Matrix — {model_name}"
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Classification reports ──
    section_header("📋 Classification Reports")
    rp1, rp2 = st.columns(2)
    with rp1:
        st.markdown("**Logistic Regression**")
        st.code(classification_report(y_test, lr_pred, target_names=["Not Readmitted", "Readmitted"]))
    with rp2:
        st.markdown("**Random Forest**")
        st.code(classification_report(y_test, rf_pred, target_names=["Not Readmitted", "Readmitted"]))

    # ── Feature importance ──
    section_header("🌟 Random Forest Feature Importance")
    st.info("ℹ️ Feature importance shows the relative contribution of each feature to the model's predictions. "
            "It reflects **statistical association, not causation**. Clinical decisions must not be made solely based on these values.")

    fi = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": res["rf"].feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        title="Feature Importance — Random Forest",
        color="Importance", color_continuous_scale="Blues",
        template="plotly_white"
    )
    fig.update_layout(height=400, coloraxis_showscale=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── ROC Curve ──
    section_header("📉 ROC Curves")
    from sklearn.metrics import roc_curve
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode="lines", name=f"LR (AUC={lr_metrics['ROC-AUC']:.3f})",
                             line=dict(color="#1e5799", width=2)))
    fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode="lines", name=f"RF (AUC={rf_metrics['ROC-AUC']:.3f})",
                             line=dict(color="#27ae60", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
                             line=dict(color="gray", width=1, dash="dash")))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        template="plotly_white", height=380,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='footer'>CareSight AI — IBM SkillsBuild Data Analytics Internship Project | Educational Use Only</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.markdown("""
    <h1 style='color:#0a2342; margin-bottom:2px;'>ℹ️ About CareSight AI</h1>
    <p style='color:#6b7280; font-size:15px; margin-top:0;'>
        IBM SkillsBuild Data Analytics with AI — Internship Capstone Project
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🏥 Project Overview
        **CareSight AI** is an interactive healthcare analytics and patient readmission intelligence
        platform built as part of the IBM SkillsBuild Data Analytics with AI Internship programme.

        The platform enables healthcare analysts to:
        - Explore patient admission patterns across conditions, demographics, and regions
        - Identify key drivers of hospital readmission using machine learning
        - Predict individual patient readmission risk interactively
        - Compare model performance with industry-standard evaluation metrics

        ---
        ### 📊 Dataset Summary
        | Attribute | Value |
        |-----------|-------|
        | Total Records | 984 patients |
        | Features | 15 columns |
        | Target Variable | Readmission (Yes/No) |
        | Year Range | 2022–2025 |
        | Medical Conditions | 14 unique |
        | States Covered | 28 Indian states |
        | Source | Supplied hospital patient dataset |

        ---
        ### 🛠️ Technology Stack
        | Component | Technology |
        |-----------|-----------|
        | Frontend / UI | Streamlit |
        | Data Processing | Pandas, NumPy |
        | Visualisation | Plotly Express & Graph Objects |
        | Machine Learning | Scikit-learn |
        | Models | Logistic Regression, Random Forest |
        | Language | Python 3.10+ |

        ---
        ### 🤖 ML Workflow
        ```
        Dataset  ──►  Data Cleaning  ──►  Feature Engineering
            ──►  EDA  ──►  Preprocessing  ──►  Train/Test Split (80/20)
            ──►  Logistic Regression & Random Forest
            ──►  Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
            ──►  Feature Importance  ──►  Interactive Predictions
        ```

        ---
        ### 📁 Application Pages
        | Page | Description |
        |------|-------------|
        | 📊 Dashboard | KPI cards, condition/readmission/outcome/cost charts |
        | 🔍 Patient Analytics | 7-filter explorer with dynamic charts |
        | 🤖 AI Insights | Interactive patient form + readmission prediction |
        | 📈 Model Performance | Confusion matrix, ROC curves, feature importance |
        | ℹ️ About Project | This page — project info, team, methodology |
        """)

    with col2:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:20px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-top:10px;'>
            <div style='font-size:15px; font-weight:700; color:#0a2342; margin-bottom:12px;'>
                👤 Project Author
            </div>
            <div style='font-size:14px; color:#374151; line-height:1.8;'>
                <b>Name:</b> Anjum<br>
                <b>Programme:</b> IBM SkillsBuild<br>
                <b>Domain:</b> Data Analytics with AI<br>
                <b>Project:</b> CareSight AI<br>
                <b>Year:</b> 2024–25
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#e8f4fd; border-radius:12px; padding:20px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-top:14px;'>
            <div style='font-size:15px; font-weight:700; color:#0a2342; margin-bottom:10px;'>
                ⚠️ Limitations
            </div>
            <ul style='font-size:13px; color:#374151; padding-left:18px; line-height:1.9;'>
                <li>Dataset is synthetic / educational</li>
                <li>Models not validated clinically</li>
                <li>No real-time data integration</li>
                <li>Limited feature engineering</li>
                <li>Class imbalance not fully addressed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#f0fdf4; border-radius:12px; padding:20px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-top:14px;'>
            <div style='font-size:15px; font-weight:700; color:#0a2342; margin-bottom:10px;'>
                🚀 Future Scope
            </div>
            <ul style='font-size:13px; color:#374151; padding-left:18px; line-height:1.9;'>
                <li>XGBoost / deep learning models</li>
                <li>SHAP explainability</li>
                <li>Real hospital EHR integration</li>
                <li>Multi-language dashboard</li>
                <li>Automated PDF report export</li>
                <li>Geospatial state-level maps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer-box' style='margin-top:20px;'>
        ⚕️ <b>Healthcare Disclaimer:</b> CareSight AI is strictly for educational and analytical
        demonstration purposes. All predictions, visualisations, and insights are derived from a
        sample dataset and <b>must not be used for real clinical, diagnostic, or treatment decisions</b>.
        Always consult a qualified medical professional for healthcare guidance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='footer'>CareSight AI — IBM SkillsBuild Data Analytics Internship Project | Educational Use Only</div>",
                unsafe_allow_html=True)
