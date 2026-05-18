"""
app/main_dashboard.py  –  Streamlit multi-page dashboard
Run: streamlit run app/main_dashboard.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.decomposition import PCA

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main { background: #0a0e1a; }
.block-container { padding: 1.5rem 2rem; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #0d1526 0%, #1a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,80,200,0.15);
}
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #4d9fff; font-family: 'IBM Plex Mono'; }
.kpi-label { font-size: 0.78rem; color: #8099bb; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
.kpi-delta { font-size: 0.85rem; margin-top: 6px; }

/* Section headers */
.section-header {
    font-size: 1.05rem; font-weight: 600; color: #7ab8ff;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #1e3a5f; padding-bottom: 8px;
    margin-bottom: 16px; margin-top: 8px;
}

/* Risk badges */
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-critical { background:#ff1a1a22; color:#ff4d4d; border:1px solid #ff4d4d44; }
.badge-high     { background:#ff800022; color:#ff9933; border:1px solid #ff993344; }
.badge-medium   { background:#ffdd0022; color:#ffcc00; border:1px solid #ffcc0044; }
.badge-low      { background:#00cc6622; color:#00dd77; border:1px solid #00dd7744; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "customers.csv")
MODEL_DIR = os.path.join(BASE, "models")

# ── Load data (cached) ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_models():
    import joblib
    try:
        model     = joblib.load(os.path.join(MODEL_DIR, "xgb_churn_model.pkl"))
        km        = joblib.load(os.path.join(MODEL_DIR, "kmeans_segments.pkl"))
        scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        le_dict   = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
        feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        return model, km, scaler, le_dict, feat_cols
    except Exception as e:
        st.error(f"Model load error: {e}. Run train_model.py first.")
        return None, None, None, None, None

@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

# ── Sidebar navigation ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2rem'>💳</div>
        <div style='font-weight:700; font-size:1rem; color:#4d9fff; letter-spacing:1px;'>CHURN INTELLIGENCE</div>
        <div style='font-size:0.7rem; color:#556b8a; margin-top:4px;'>Enterprise Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "📊  Dataset Explorer",
        "🔮  Churn Predictor",
        "🎯  Segmentation",
        "📈  Business Analytics",
        "🤖  AI Retention Advisor"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#556b8a;text-align:center'>v1.0.0 · Built for AmEx / JPMorgan</div>", unsafe_allow_html=True)


df = load_data()
model, km, scaler, le_dict, feat_cols = load_models()
metrics = load_metrics()

# ════════════════════════════════════════════════════════════════════
#  PAGE 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("## 🏠 Customer Churn Intelligence — Overview")
    st.markdown("<div style='color:#7ab8ff;margin-bottom:24px'>Real-time analytics · 5,000 credit card customers · XGBoost + KMeans</div>", unsafe_allow_html=True)

    # KPI row
    churn_rate = df["churn_label"].mean()
    total = len(df)
    churned = df["churn_label"].sum()
    retained = total - churned
    avg_spend = df["total_trans_amount"].mean()
    at_risk = df[df["churn_label"]==1]["total_trans_amount"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{total:,}", "Total Customers", ""),
        (f"{churn_rate:.1%}", "Churn Rate", "🔴"),
        (f"{churned:,}", "Churned Customers", ""),
        (f"${avg_spend:,.0f}", "Avg Annual Spend", ""),
        (f"${at_risk/1e6:.1f}M", "Revenue at Risk", "🚨"),
    ]
    for col, (val, label, icon) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{icon} {val}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Churn Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        vals = df["churn_flag"].value_counts()
        colors = ["#4d9fff", "#ff4d4d"]
        wedges, texts, autotexts = ax.pie(
            vals.values, labels=vals.index, autopct="%1.1f%%",
            colors=colors, startangle=90,
            wedgeprops=dict(edgecolor="#0a0e1a", linewidth=2)
        )
        for t in texts + autotexts:
            t.set_color("white"); t.set_fontsize(11)
        ax.set_title("Customer Status", color="white", fontsize=13, pad=10)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-header'>Churn by Card Type</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        gb = df.groupby("card_type")["churn_label"].mean().sort_values()
        bars = ax.barh(gb.index, gb.values*100, color=["#4d9fff","#7ab8ff","#ff9933","#ff4d4d"])
        ax.set_xlabel("Churn Rate (%)", color="#7ab8ff")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
        ax.set_title("Churn Rate by Card Type", color="white", fontsize=13)
        for bar, val in zip(bars, gb.values*100):
            ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va='center', color='white', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-header'>Monthly Spending Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        for flag, color in [("Existing","#4d9fff"),("Churned","#ff4d4d")]:
            ax.hist(df[df["churn_flag"]==flag]["total_trans_amount"], bins=40,
                    alpha=0.6, color=color, label=flag, edgecolor="none")
        ax.set_xlabel("Annual Spend ($)", color="#7ab8ff")
        ax.set_ylabel("Count", color="#7ab8ff")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
        ax.legend(facecolor="#0d1526", edgecolor="#1e3a5f", labelcolor="white")
        ax.set_title("Spend: Churned vs Existing", color="white", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("<div class='section-header'>Risk Factors — Correlation</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        corr_cols = ["churn_label","months_inactive_12m","contacts_count_12m",
                     "late_payments","utilization_ratio","total_trans_count"]
        corr = df[corr_cols].corr()["churn_label"].drop("churn_label").sort_values()
        colors = ["#ff4d4d" if v > 0 else "#4d9fff" for v in corr.values]
        ax.barh(corr.index, corr.values, color=colors)
        ax.axvline(0, color="#556b8a", linestyle="--", lw=1)
        ax.set_xlabel("Correlation with Churn", color="#7ab8ff")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
        ax.set_title("Feature Correlations", color="white", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    # Model metrics banner
    if metrics:
        st.markdown("---")
        st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, (label, val) in zip(
            [mc1, mc2, mc3, mc4],
            [("Accuracy", f"{metrics.get('accuracy',0):.2%}"),
             ("ROC-AUC",  f"{metrics.get('roc_auc',0):.4f}"),
             ("Algorithm", "XGBoost"),
             ("Training Set", "4,000 rows")]
        ):
            col.metric(label, val)


# ════════════════════════════════════════════════════════════════════
#  PAGE 2 – DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("## 📊 Dataset Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Features", f"{len(df.columns)}")
    col3.metric("Churn Rate", f"{df['churn_label'].mean():.2%}")

    st.markdown("---")
    # Filters
    st.markdown("<div class='section-header'>Filters</div>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        card_filter = st.multiselect("Card Type", df["card_type"].unique(), default=list(df["card_type"].unique()))
    with fc2:
        income_filter = st.multiselect("Income Bracket", df["income_bracket"].unique(), default=list(df["income_bracket"].unique()))
    with fc3:
        churn_filter = st.selectbox("Churn Status", ["All","Existing","Churned"])

    mask = df["card_type"].isin(card_filter) & df["income_bracket"].isin(income_filter)
    if churn_filter != "All":
        mask &= df["churn_flag"] == churn_filter
    filtered = df[mask]
    st.markdown(f"**{len(filtered):,} customers** match filters")
    st.dataframe(filtered.head(200), use_container_width=True, height=380)

    st.markdown("---")
    st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
    num_cols = ["age","credit_limit","total_trans_amount","total_trans_count","utilization_ratio","months_inactive_12m"]
    st.dataframe(filtered[num_cols].describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 3 – CHURN PREDICTOR
# ════════════════════════════════════════════════════════════════════
elif "Predictor" in page:
    st.markdown("## 🔮 Real-Time Churn Predictor")

    if model is None:
        st.error("Models not loaded. Run `python train_model.py` first.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-header'>Customer Profile Input</div>", unsafe_allow_html=True)
        age             = st.slider("Age", 21, 75, 42)
        months_on_book  = st.slider("Months as Customer", 6, 120, 36)
        credit_limit    = st.number_input("Credit Limit ($)", 1000, 35000, 8000, step=500)
        revolving_bal   = st.number_input("Revolving Balance ($)", 0, int(credit_limit), 500, step=100)
        utilization     = round(revolving_bal / max(credit_limit, 1), 4)
        st.metric("Utilization Ratio", f"{utilization:.2%}")

        total_trans_amt = st.number_input("Annual Transaction Amount ($)", 500, 25000, 4000, step=100)
        total_trans_ct  = st.slider("Annual Transaction Count", 10, 140, 60)
        months_inactive = st.slider("Months Inactive (12m)", 0, 6, 1)
        contacts        = st.slider("Support Contacts (12m)", 0, 6, 1)
        late_payments   = st.slider("Late Payments", 0, 5, 0)
        rewards         = st.slider("Rewards Redemptions", 0, 20, 5)
        logins          = st.slider("Digital Logins (30d)", 0, 40, 10)
        amt_change      = st.slider("Spend Change Q4→Q1", 0.0, 3.0, 0.75)
        ct_change       = st.slider("Transaction Count Change Q4→Q1", 0.0, 2.5, 0.70)
        dependents      = st.slider("Dependents", 0, 5, 1)

        gender          = st.selectbox("Gender", ["Male","Female"])
        income_bracket  = st.selectbox("Income Bracket", ["<$40K","$40K-$60K","$60K-$80K","$80K-$120K","$120K+"])
        education       = st.selectbox("Education", ["High School","Some College","Graduate","Post-Graduate","Uneducated"])
        marital         = st.selectbox("Marital Status", ["Married","Single","Divorced"])
        card_type       = st.selectbox("Card Type", ["Blue","Silver","Gold","Platinum"])

    with col2:
        st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)

        customer_dict = {
            "age": age, "months_on_book": months_on_book, "credit_limit": credit_limit,
            "revolving_balance": revolving_bal, "avg_open_to_buy": credit_limit - revolving_bal,
            "utilization_ratio": utilization, "total_trans_amount": total_trans_amt,
            "total_trans_count": total_trans_ct, "avg_trans_amount": total_trans_amt / max(total_trans_ct,1),
            "total_amt_change_q4_q1": amt_change, "total_ct_change_q4_q1": ct_change,
            "months_inactive_12m": months_inactive, "contacts_count_12m": contacts,
            "late_payments": late_payments, "rewards_redemptions": rewards,
            "digital_logins_30d": logins, "dependent_count": dependents,
            "gender": gender, "income_bracket": income_bracket,
            "education_level": education, "marital_status": marital, "card_type": card_type
        }

        from churn_prediction import predict_single
        pred = predict_single(customer_dict)

        prob = pred["churn_probability"]
        risk = pred["risk_level"]
        color_map = {"Critical":"#ff4d4d","High":"#ff9933","Medium":"#ffcc00","Low":"#00dd77"}
        col = color_map.get(risk, "#4d9fff")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d1526,#1a2540);border:2px solid {col};
             border-radius:16px;padding:28px;text-align:center;margin-bottom:16px;'>
            <div style='font-size:3.5rem;font-weight:700;color:{col};font-family:IBM Plex Mono'>
                {prob:.1%}
            </div>
            <div style='font-size:1rem;color:#aac4e0;margin-top:4px'>Churn Probability</div>
            <div style='margin-top:12px'>
                <span class='badge badge-{"critical" if risk=="Critical" else risk.lower()}'>{risk} Risk</span>
                &nbsp;
                <span style='color:#7ab8ff;font-size:0.9rem'>{pred["segment"]}</span>
            </div>
            <div style='margin-top:16px;font-size:1.1rem;color:{"#ff4d4d" if pred["churn_label"] else "#00dd77"}'>
                {'⚠️ LIKELY TO CHURN' if pred["churn_label"] else '✅ LIKELY TO STAY'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        ax.barh(["Churn Risk"], [prob], color=col, height=0.4)
        ax.barh(["Churn Risk"], [1-prob], left=[prob], color="#1e3a5f", height=0.4)
        ax.set_xlim(0,1); ax.set_xticks([0,0.25,0.5,0.75,1.0])
        ax.set_xticklabels(["0%","25%","50%","75%","100%"], color="white")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
        ax.set_title(f"Risk Score: {prob:.1%}", color="white", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        # Feature importance for this prediction
        if metrics.get("feature_importance"):
            st.markdown("<div class='section-header' style='margin-top:20px'>Key Risk Drivers</div>", unsafe_allow_html=True)
            fi = metrics["feature_importance"]
            top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0d1526")
            ax.set_facecolor("#0d1526")
            names = [n.replace("_enc","").replace("_"," ").title() for n,_ in top]
            vals2 = [v for _,v in top]
            ax.barh(names[::-1], vals2[::-1], color="#4d9fff")
            ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
            ax.set_xlabel("Importance", color="#7ab8ff")
            fig.tight_layout()
            st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════
#  PAGE 4 – SEGMENTATION
# ════════════════════════════════════════════════════════════════════
elif "Segment" in page:
    st.markdown("## 🎯 Customer Segmentation")

    if model is None:
        st.error("Models not loaded."); st.stop()

    from churn_prediction import predict_batch
    @st.cache_data
    def get_segmented():
        return predict_batch(df.copy())

    seg_df = get_segmented()

    # Segment KPIs
    seg_counts = seg_df["segment"].value_counts()
    st.markdown("<div class='section-header'>Segment Overview</div>", unsafe_allow_html=True)
    cols = st.columns(len(seg_counts))
    seg_colors = {"High Value":"#4d9fff","Low Engagement":"#ff9933","Loyal Customer":"#00dd77","At-Risk":"#ff4d4d"}
    for c, (seg, cnt) in zip(cols, seg_counts.items()):
        churn_pct = seg_df[seg_df["segment"]==seg]["churn_label"].mean()
        c.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value' style='color:{seg_colors.get(seg,"#4d9fff")};font-size:1.8rem'>{cnt:,}</div>
            <div class='kpi-label'>{seg}</div>
            <div style='color:#ff9933;font-size:0.8rem;margin-top:6px'>{churn_pct:.1%} churn rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Segment Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        sc = seg_df["segment"].value_counts()
        colors_list = [seg_colors.get(s,"#4d9fff") for s in sc.index]
        ax.bar(sc.index, sc.values, color=colors_list)
        ax.set_ylabel("Count", color="#7ab8ff")
        ax.tick_params(colors="white", rotation=15); ax.spines[:].set_color("#1e3a5f")
        ax.set_title("Customers per Segment", color="white", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-header'>Avg Spend by Segment</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        seg_spend = seg_df.groupby("segment")["total_trans_amount"].mean().sort_values(ascending=False)
        colors_list2 = [seg_colors.get(s,"#4d9fff") for s in seg_spend.index]
        bars = ax.bar(seg_spend.index, seg_spend.values, color=colors_list2)
        ax.set_ylabel("Avg Annual Spend ($)", color="#7ab8ff")
        ax.tick_params(colors="white", rotation=15); ax.spines[:].set_color("#1e3a5f")
        ax.set_title("Average Spend by Segment", color="white", fontsize=13)
        for bar, val in zip(bars, seg_spend.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+50, f"${val:,.0f}", ha='center', color='white', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    # PCA scatter
    st.markdown("<div class='section-header'>2D Cluster Visualization (PCA)</div>", unsafe_allow_html=True)
    import joblib as jl
    sc2 = jl.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    fc2 = jl.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    CAT_FEATURES = ["gender","income_bracket","education_level","marital_status","card_type"]
    df2 = df.copy()
    for col_n in CAT_FEATURES:
        le = le_dict[col_n]
        df2[col_n+"_enc"] = df2[col_n].astype(str).apply(
            lambda v: int(le.transform([v])[0]) if v in le.classes_ else 0
        )
    X_all = df2[fc2].fillna(0)
    X_scaled = sc2.transform(X_all)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    segs = seg_df["segment"].values
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1526")
    ax.set_facecolor("#0d1526")
    for seg_name, sc_col in seg_colors.items():
        mask_s = segs == seg_name
        ax.scatter(coords[mask_s, 0], coords[mask_s, 1], c=sc_col,
                   label=seg_name, alpha=0.5, s=8)
    ax.legend(facecolor="#0d1526", edgecolor="#1e3a5f", labelcolor="white")
    ax.set_title("Customer Segments (PCA)", color="white", fontsize=13)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
    ax.set_xlabel("PC1", color="#7ab8ff"); ax.set_ylabel("PC2", color="#7ab8ff")
    fig.tight_layout()
    st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════
#  PAGE 5 – BUSINESS ANALYTICS
# ════════════════════════════════════════════════════════════════════
elif "Business" in page:
    st.markdown("## 📈 Business Analytics")

    # Ensure SQL DB exists
    import sqlite3
    DB_PATH = os.path.join(BASE, "data", "churn_analytics.db")
    if not os.path.exists(DB_PATH):
        con = sqlite3.connect(DB_PATH)
        df.to_sql("customers", con, if_exists="replace", index=False)
        con.close()

    def sql_query(q):
        con = sqlite3.connect(DB_PATH)
        r = pd.read_sql_query(q, con)
        con.close()
        return r

    st.markdown("<div class='section-header'>Churn Rate by Card Type (SQL)</div>", unsafe_allow_html=True)
    q1 = sql_query("""
        SELECT card_type,COUNT(*) total,SUM(churn_label) churned,
               ROUND(100.0*SUM(churn_label)/COUNT(*),2) churn_pct,
               ROUND(AVG(total_trans_amount),2) avg_spend
        FROM customers GROUP BY card_type ORDER BY churn_pct DESC
    """)
    st.dataframe(q1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Revenue at Risk by Income</div>", unsafe_allow_html=True)
        q2 = sql_query("""
            SELECT income_bracket,
                   ROUND(SUM(CASE WHEN churn_label=1 THEN total_trans_amount ELSE 0 END),0) revenue_at_risk,
                   COUNT(*) customers
            FROM customers GROUP BY income_bracket ORDER BY revenue_at_risk DESC
        """)
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        ax.barh(q2["income_bracket"], q2["revenue_at_risk"]/1000, color="#ff4d4d")
        ax.set_xlabel("Revenue at Risk ($K)", color="#7ab8ff")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
        ax.set_title("Revenue at Risk by Income", color="white")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-header'>Churn by Months Inactive</div>", unsafe_allow_html=True)
        q3 = sql_query("""
            SELECT months_inactive_12m,
                   ROUND(100.0*SUM(churn_label)/COUNT(*),1) churn_pct,
                   COUNT(*) customers
            FROM customers GROUP BY months_inactive_12m ORDER BY months_inactive_12m
        """)
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        ax.plot(q3["months_inactive_12m"], q3["churn_pct"], "o-", color="#ff4d4d", lw=2)
        ax.fill_between(q3["months_inactive_12m"], q3["churn_pct"], alpha=0.15, color="#ff4d4d")
        ax.set_xlabel("Months Inactive", color="#7ab8ff")
        ax.set_ylabel("Churn Rate (%)", color="#7ab8ff")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#1e3a5f")
        ax.set_title("Inactivity vs Churn", color="white")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Top High-Value Customers at Risk</div>", unsafe_allow_html=True)
    q4 = sql_query("""
        SELECT customer_id, card_type,
               ROUND(total_trans_amount,0) spend,
               months_inactive_12m, contacts_count_12m, late_payments, churn_flag
        FROM customers WHERE churn_label=1
        ORDER BY total_trans_amount DESC LIMIT 15
    """)
    st.dataframe(q4, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-header'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
    heat_cols = ["churn_label","age","credit_limit","utilization_ratio","total_trans_amount",
                 "total_trans_count","months_inactive_12m","contacts_count_12m","late_payments",
                 "rewards_redemptions","digital_logins_30d"]
    fig, ax = plt.subplots(figsize=(10,6), facecolor="#0d1526")
    corr = df[heat_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                annot_kws={"size":7}, linewidths=0.3, linecolor="#0a0e1a")
    ax.tick_params(colors="white",labelsize=8)
    ax.set_title("Correlation Matrix", color="white", fontsize=13, pad=12)
    fig.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════
#  PAGE 6 – AI RETENTION ADVISOR
# ════════════════════════════════════════════════════════════════════
elif "AI" in page:
    st.markdown("## 🤖 AI-Powered Retention Advisor")
    st.markdown("<div style='color:#7ab8ff;margin-bottom:16px'>GenAI retention strategies · Powered by Claude AI (falls back to rule-based engine)</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='section-header'>Customer Profile</div>", unsafe_allow_html=True)
        # Quick-select from dataset
        sample_id = st.selectbox("Load Sample Customer", df[df["churn_label"]==1]["customer_id"].head(20).tolist())
        sample_row = df[df["customer_id"]==sample_id].iloc[0].to_dict()

        age          = st.number_input("Age", 21, 75, int(sample_row.get("age",42)))
        card_type    = st.selectbox("Card Type", ["Blue","Silver","Gold","Platinum"],
                                    index=["Blue","Silver","Gold","Platinum"].index(sample_row.get("card_type","Blue")))
        income       = st.selectbox("Income", ["<$40K","$40K-$60K","$60K-$80K","$80K-$120K","$120K+"])
        spend        = st.number_input("Annual Spend ($)", 500, 25000, int(sample_row.get("total_trans_amount",3000)))
        inactive     = st.slider("Months Inactive", 0, 6, int(sample_row.get("months_inactive_12m",3)))
        contacts     = st.slider("Support Contacts", 0, 6, int(sample_row.get("contacts_count_12m",3)))
        late         = st.slider("Late Payments", 0, 5, int(sample_row.get("late_payments",1)))
        rewards      = st.slider("Rewards Redemptions", 0, 20, int(sample_row.get("rewards_redemptions",1)))
        logins       = st.slider("Digital Logins", 0, 40, int(sample_row.get("digital_logins_30d",4)))
        credit_lim   = st.number_input("Credit Limit", 1000, 35000, int(sample_row.get("credit_limit",8000)))
        util         = st.number_input("Utilization Ratio", 0.0, 1.0,
                                       float(round(sample_row.get("utilization_ratio",0.08),2)), step=0.01)

        generate_btn = st.button("🚀 Generate AI Retention Strategy", use_container_width=True, type="primary")

    with col2:
        st.markdown("<div class='section-header'>AI Retention Strategy</div>", unsafe_allow_html=True)

        if generate_btn:
            customer = {
                "age": age, "gender": "Unknown", "card_type": card_type,
                "months_on_book": 36, "income_bracket": income,
                "education_level": "Graduate", "credit_limit": credit_lim,
                "utilization_ratio": util, "total_trans_amount": spend,
                "total_trans_count": 45, "months_inactive_12m": inactive,
                "contacts_count_12m": contacts, "late_payments": late,
                "rewards_redemptions": rewards, "digital_logins_30d": logins,
                "total_amt_change_q4_q1": 0.6, "total_ct_change_q4_q1": 0.55,
                "avg_open_to_buy": credit_lim * (1-util),
                "revolving_balance": credit_lim * util,
                "avg_trans_amount": spend/45, "dependent_count": 1,
                "marital_status": "Married"
            }
            from churn_prediction import predict_single
            from retention_ai import generate_retention_strategy

            with st.spinner("Analyzing customer profile..."):
                pred     = predict_single(customer)
                strategy = generate_retention_strategy(customer, pred)

            prob  = pred["churn_probability"]
            risk  = pred["risk_level"]
            cmap  = {"Critical":"#ff4d4d","High":"#ff9933","Medium":"#ffcc00","Low":"#00dd77"}
            c     = cmap.get(risk,"#4d9fff")

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d1526,#1a2540);
                 border:2px solid {c};border-radius:12px;padding:20px;margin-bottom:16px'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div>
                        <div style='font-size:0.75rem;color:#8099bb;text-transform:uppercase;letter-spacing:1px'>Churn Probability</div>
                        <div style='font-size:2.4rem;font-weight:700;color:{c};font-family:IBM Plex Mono'>{prob:.1%}</div>
                    </div>
                    <div style='text-align:right'>
                        <span class='badge badge-{"critical" if risk=="Critical" else risk.lower()}'>{risk} Risk</span><br/>
                        <span style='color:#7ab8ff;font-size:0.85rem;margin-top:6px;display:inline-block'>{pred["segment"]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#0d1526;border:1px solid #1e3a5f;border-radius:10px;padding:16px;margin-bottom:12px'>
                <div style='color:#4d9fff;font-weight:600;margin-bottom:8px'>🎯 {strategy.get("strategy_title","Retention Strategy")}</div>
                <div style='color:#aac4e0;font-size:0.9rem'>{strategy.get("churn_reason","")}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**📋 Immediate Actions:**")
            for action in strategy.get("immediate_actions", []):
                st.markdown(f"- {action}")

            st.markdown(f"""
            <div style='background:#0a1428;border-left:3px solid #4d9fff;padding:12px 16px;
                 border-radius:0 8px 8px 0;margin:12px 0'>
                <div style='color:#7ab8ff;font-size:0.75rem;text-transform:uppercase'>Customer Message</div>
                <div style='color:#cce0ff;font-style:italic;margin-top:4px'>"{strategy.get('customer_message','')}"</div>
            </div>
            """, unsafe_allow_html=True)

            cols2 = st.columns(3)
            cols2[0].metric("Best Channel", strategy.get("communication_channel","Email"))
            cols2[1].metric("Expected Lift", strategy.get("expected_retention_lift","–"))
            cols2[2].metric("Strategy Source", strategy.get("source","Rule-Based"))

        else:
            st.markdown("""
            <div style='background:#0d1526;border:1px dashed #1e3a5f;border-radius:12px;
                 padding:40px;text-align:center;color:#556b8a'>
                <div style='font-size:2rem'>🤖</div>
                <div style='margin-top:8px'>Select a customer and click <b>Generate</b> to get AI-powered retention recommendations</div>
            </div>
            """, unsafe_allow_html=True)
