import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Creator DSS",
    page_icon="🎥",
    layout="wide"
)

# ── Load models and data ──────────────────────────────────
@st.cache_resource
def load_models():
    from xgboost import XGBClassifier
    import pickle

    # Load XGBoost from JSON
    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODELS / "xgb_model.json"))

    # Load encoders
    with open(MODELS / "le_target.pkl", "rb") as f: le_target = pickle.load(f)
    with open(MODELS / "le_cat.pkl", "rb") as f: le_cat    = pickle.load(f)

    return xgb_model, le_target, le_cat

@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "youtube_final.csv")
    return df

xgb_model, le_target, le_cat = load_models()
df = load_data()

# ── Feature columns ───────────────────────────────────────
feature_cols = [
    "likes", "dislikes", "comment_count",
    "days_to_trending", "tag_count", "title_length",
    "publish_hour", "comments_disabled", "ratings_disabled",
    "engagement_rate", "like_ratio", "dislike_ratio",
    "comment_ratio", "publish_day_num", "category_encoded"
]

day_map = {
    "Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
    "Friday":4,"Saturday":5,"Sunday":6
}

# ── Compute benchmarks ────────────────────────────────────
@st.cache_data
def get_benchmarks(df):
    benchmarks = df.groupby("category_name").agg(
        avg_views         = ("view_count",      "mean"),
        avg_engagement    = ("engagement_rate", "mean"),
        avg_like_ratio    = ("like_ratio",      "mean"),
        avg_tag_count     = ("tag_count",       "mean"),
        avg_title_length  = ("title_length",    "mean"),
        avg_days_trending = ("days_to_trending","mean"),
        high_pct          = ("success_class",   lambda x: (x=="High").mean()*100)
    ).round(4)

    best_day = df.groupby(["category_name","publish_day"])["view_count"]\
                 .mean().reset_index()
    best_day = best_day.loc[best_day.groupby("category_name")["view_count"]\
                                    .idxmax()][["category_name","publish_day"]]
    best_day.columns = ["category_name","best_publish_day"]

    best_hour = df.groupby(["category_name","publish_hour"])["view_count"]\
                  .mean().reset_index()
    best_hour = best_hour.loc[best_hour.groupby("category_name")["view_count"]\
                                       .idxmax()][["category_name","publish_hour"]]
    best_hour.columns = ["category_name","best_publish_hour"]

    benchmarks = benchmarks.reset_index()
    benchmarks = benchmarks.merge(best_day,  on="category_name")
    benchmarks = benchmarks.merge(best_hour, on="category_name")
    benchmarks = benchmarks.set_index("category_name")
    return benchmarks

benchmarks = get_benchmarks(df)

# ════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=150)
st.sidebar.title("YouTube Creator DSS")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to:", [
    "🏠 Overview",
    "🔮 Success Predictor",
    "📊 Analytics Dashboard",
    "💡 Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 47,076 US Trending Videos")
st.sidebar.markdown("**Model:** XGBoost (98.95% accuracy)")
st.sidebar.markdown("**Categories:** 15 content types")

# ════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🎥 YouTube Creator Decision Support System")
    st.markdown("*A data-driven framework for predicting video success and generating actionable content strategy recommendations*")
    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Videos Analysed", "47,076", "US Trending Dataset")
    col2.metric("Model Accuracy",        "98.95%", "XGBoost Classifier")
    col3.metric("Avg Views",             "2.50M",  "Per video")
    col4.metric("Avg Engagement Rate",   "5.10%",  "Likes + Comments / Views")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Success Class Distribution")
        class_counts = df["success_class"].value_counts().reindex(["Low","Medium","High"])
        colors = ["#E63946","#E9C46A","#2A9D8F"]
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor="white")
        for bar, v in zip(bars, class_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                    f"{v:,}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Number of Videos")
        ax.set_title("Videos by Success Class")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🏆 Top 5 Categories by Avg Views")
        cat_views = df.groupby("category_name")["view_count"]\
                      .mean().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(cat_views.index, cat_views.values/1e6,
                color="#534AB7", edgecolor="white")
        ax.set_xlabel("Avg Views (Millions)")
        ax.set_title("Top Categories by Performance")
        for i, v in enumerate(cat_views.values/1e6):
            ax.text(v+0.05, i, f"{v:.2f}M", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🔑 Key Findings")
    findings = [
        ("🎵 Music dominates", "4.58M avg views — highest of all 15 categories"),
        ("📅 Friday is optimal", "3.20M avg views and 36.5% High performance rate"),
        ("⏰ 9AM is peak hour", "8.66M avg views — best upload time across dataset"),
        ("👍 Likes drive success", "0.78 correlation with views — strongest predictor"),
        ("📊 Engagement rate ≠ success", "Low/Medium/High all average ~5.1% engagement"),
        ("🚀 XGBoost wins", "98.95% accuracy with AUC-ROC of 0.9999"),
    ]
    col1, col2 = st.columns(2)
    for i, (title, desc) in enumerate(findings):
        col = col1 if i % 2 == 0 else col2
        col.success(f"**{title}:** {desc}")

# ════════════════════════════════════════════════════════
# PAGE 2 — SUCCESS PREDICTOR
# ════════════════════════════════════════════════════════
elif page == "🔮 Success Predictor":
    st.title("🔮 YouTube Video Success Predictor")
    st.markdown("Enter your video details below to predict its performance tier")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Video Details")
        category    = st.selectbox("Content Category", sorted(df["category_name"].unique()))
        publish_day  = st.selectbox("Publish Day", list(day_map.keys()))
        publish_hour = st.slider("Publish Hour (24hr format)", 0, 23, 9)
        title_length = st.slider("Title Length (words)", 1, 25, 8)
        tag_count    = st.slider("Number of Tags", 0, 78, 15)

    with col2:
        st.subheader("📊 Engagement Metrics")
        likes           = st.number_input("Expected Likes",         min_value=0, value=5000,  step=500)
        comment_count   = st.number_input("Expected Comments",      min_value=0, value=300,   step=50)
        engagement_rate = st.number_input("Engagement Rate (%)",    min_value=0.0, value=5.0, step=0.1)
        dislikes        = st.number_input("Expected Dislikes",      min_value=0, value=100,   step=10)

    st.markdown("---")

    if st.button("🚀 Predict Performance", use_container_width=True):
        # Prepare input
        cat_encoded  = le_cat.transform([category])[0]
        like_ratio   = likes / max(likes * 10, 1)
        dislike_ratio= dislikes / max(likes * 10, 1)
        comment_ratio= comment_count / max(likes * 10, 1)

        input_data = pd.DataFrame([[
            likes, dislikes, comment_count,
            5.5, tag_count, title_length,
            publish_hour, 0, 0,
            engagement_rate, like_ratio, dislike_ratio,
            comment_ratio, day_map[publish_day], cat_encoded
        ]], columns=feature_cols)

        pred_class  = xgb_model.predict(input_data)[0]
        pred_label  = le_target.inverse_transform([pred_class])[0]
        pred_proba  = xgb_model.predict_proba(input_data)[0]
        confidence  = max(pred_proba) * 100

        # Show result
        col1, col2, col3 = st.columns(3)

        color_map = {"High":"🟢", "Medium":"🟡", "Low":"🔴"}
        col1.metric("Predicted Class",
                    f"{color_map.get(pred_label,'')} {pred_label}",
                    f"Confidence: {confidence:.1f}%")

        bench = benchmarks.loc[category]
        col2.metric("Category Benchmark Views",
                    f"{bench['avg_views']/1e6:.2f}M",
                    f"{bench['high_pct']:.1f}% High in {category}")

        col3.metric("Recommended Upload",
                    f"{bench['best_publish_day']}",
                    f"Best hour: {int(bench['best_publish_hour']):02d}:00")

        # Probability bar chart
        st.markdown("### Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(8, 2))
        classes = le_target.classes_
        colors  = []
        for c in classes:
            if c == "High":   colors.append("#2A9D8F")
            elif c == "Low":  colors.append("#E63946")
            else:             colors.append("#E9C46A")

        bars = ax.barh(classes, pred_proba * 100,
                       color=colors, edgecolor="white", height=0.5)
        for bar, v in zip(bars, pred_proba * 100):
            ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 110)
        ax.set_title("Performance Class Probabilities")
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS DASHBOARD
# ════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    selected_cat = st.selectbox("Filter by Category",
                                ["All"] + sorted(df["category_name"].unique()))
    filtered_df  = df if selected_cat == "All" else df[df["category_name"] == selected_cat]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Videos",          f"{len(filtered_df):,}")
    col2.metric("Avg Views",       f"{filtered_df['view_count'].mean()/1e6:.2f}M")
    col3.metric("Avg Engagement",  f"{filtered_df['engagement_rate'].mean():.2f}%")
    col4.metric("High %",          f"{(filtered_df['success_class']=='High').mean()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Avg Views by Publish Day")
        day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_views  = filtered_df.groupby("publish_day")["view_count"].mean().reindex(day_order)
        colors_day = ["#534AB7" if d == "Friday" else "#DDE3EC" for d in day_order]
        fig, ax    = plt.subplots(figsize=(6, 3))
        ax.bar(day_views.index, day_views.values/1e6, color=colors_day, edgecolor="white")
        ax.set_ylabel("Avg Views (M)")
        ax.set_title("Views by Publish Day")
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("⏰ Avg Views by Publish Hour")
        hour_views  = filtered_df.groupby("publish_hour")["view_count"].mean()
        colors_hour = ["#E07B39" if h == hour_views.idxmax() else "#DDE3EC"
                       for h in hour_views.index]
        fig, ax     = plt.subplots(figsize=(6, 3))
        ax.bar(hour_views.index, hour_views.values/1e6, color=colors_hour, edgecolor="white")
        ax.set_xlabel("Hour (24hr)")
        ax.set_ylabel("Avg Views (M)")
        ax.set_title("Views by Publish Hour")
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ Engagement Rate by Category")
        cat_eng = filtered_df.groupby("category_name")["engagement_rate"]\
                             .mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(cat_eng.index, cat_eng.values,
                color="#2A9D8F", edgecolor="white")
        ax.set_xlabel("Avg Engagement Rate (%)")
        ax.set_title("Engagement by Category")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🎯 Success Class by Category")
        cat_success = filtered_df.groupby(["category_name","success_class"])\
                                  .size().unstack(fill_value=0)
        cat_success = cat_success.reindex(columns=["Low","Medium","High"], fill_value=0)
        cat_pct     = cat_success.div(cat_success.sum(axis=1), axis=0) * 100
        fig, ax     = plt.subplots(figsize=(6, 4))
        cat_pct.plot(kind="barh", ax=ax,
                     color=["#E63946","#E9C46A","#2A9D8F"],
                     edgecolor="white")
        ax.set_xlabel("Percentage (%)")
        ax.set_title("Success Class % by Category")
        ax.legend(title="Class", loc="lower right")
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════
elif page == "💡 Recommendations":
    st.title("💡 Creator Recommendation Engine")
    st.markdown("Get personalised recommendations based on your video details and category benchmarks")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        category     = st.selectbox("Your Content Category", sorted(df["category_name"].unique()))
        publish_day  = st.selectbox("Your Publish Day",  list(day_map.keys()))
        publish_hour = st.slider("Your Publish Hour", 0, 23, 14)

    with col2:
        tag_count       = st.slider("Your Tag Count",        0, 78, 10)
        title_length    = st.slider("Your Title Length (words)", 1, 25, 6)
        engagement_rate = st.number_input("Your Engagement Rate (%)", 0.0, 40.0, 4.5, 0.1)

    likes         = st.number_input("Your Likes", 0, 10000000, 5000, 500)
    comment_count = st.number_input("Your Comments", 0, 1000000, 300, 50)

    if st.button("💡 Get My Recommendations", use_container_width=True):
        bench = benchmarks.loc[category]

        st.markdown("---")
        st.subheader(f"📋 Recommendations for {category} Creator")

        # Predict
        cat_encoded = le_cat.transform([category])[0]
        like_ratio  = likes / max(likes * 10, 1)
        input_data  = pd.DataFrame([[
            likes, 0, comment_count, 5.5,
            tag_count, title_length, publish_hour, 0, 0,
            engagement_rate, like_ratio, 0,
            comment_count / max(likes * 10, 1),
            day_map[publish_day], cat_encoded
        ]], columns=feature_cols)

        pred_class = xgb_model.predict(input_data)[0]
        pred_label = le_target.inverse_transform([pred_class])[0]
        pred_proba = xgb_model.predict_proba(input_data)[0]
        confidence = max(pred_proba) * 100

        color_map = {"High":"🟢", "Medium":"🟡", "Low":"🔴"}
        st.metric("Predicted Performance",
                  f"{color_map.get(pred_label,'')} {pred_label}",
                  f"Confidence: {confidence:.1f}%")

        st.markdown("---")
        st.subheader("📊 Your Metrics vs Category Benchmark")

        metrics_data = {
            "Metric":     ["Engagement Rate %", "Tag Count", "Title Length (words)"],
            "Your Value": [engagement_rate, tag_count, title_length],
            "Benchmark":  [round(bench["avg_engagement"], 2),
                           round(bench["avg_tag_count"]),
                           round(bench["avg_title_length"], 1)],
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df["Status"] = metrics_df.apply(
            lambda r: "✅ Above" if r["Your Value"] >= r["Benchmark"] else "⚠️ Below", axis=1
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("⏰ Timing Recommendations")
        col1, col2 = st.columns(2)
        col1.metric("Best Publish Day for " + category,
                    bench["best_publish_day"],
                    "Your day: " + publish_day +
                    (" ✅" if publish_day == bench["best_publish_day"] else " ⚠️ Change"))
        col2.metric("Best Publish Hour for " + category,
                    f"{int(bench['best_publish_hour']):02d}:00",
                    f"Your hour: {publish_hour:02d}:00" +
                    (" ✅" if publish_hour == bench["best_publish_hour"] else " ⚠️ Change"))

        st.markdown("---")
        st.subheader("💡 Specific Action Items")
        recs = []

        if engagement_rate < bench["avg_engagement"]:
            recs.append(f"📌 Increase engagement rate above {bench['avg_engagement']:.2f}% — add calls-to-action in your video")
        if tag_count < bench["avg_tag_count"]:
            recs.append(f"🏷️ Add more tags — you have {tag_count}, benchmark is {int(bench['avg_tag_count'])} tags")
        if tag_count > bench["avg_tag_count"] * 1.5:
            recs.append(f"✂️ Reduce tags — target around {int(bench['avg_tag_count'])} tags for this category")
        if title_length < bench["avg_title_length"]:
            recs.append(f"📝 Lengthen your title — target {bench['avg_title_length']:.0f} words (current: {title_length})")
        if title_length > bench["avg_title_length"] * 1.3:
            recs.append(f"✂️ Shorten your title — target around {bench['avg_title_length']:.0f} words")
        if publish_day != bench["best_publish_day"]:
            recs.append(f"📅 Switch to {bench['best_publish_day']} — best day for {category} category")
        if publish_hour != bench["best_publish_hour"]:
            recs.append(f"⏰ Switch to {int(bench['best_publish_hour']):02d}:00 — best hour for {category} category")
        if not recs:
            recs.append("✅ Your video is well optimised for this category — keep it up!")

        for rec in recs:
            st.info(rec)

        st.markdown("---")
        st.subheader("🎯 Category Insight")
        col1, col2, col3 = st.columns(3)
        col1.metric("High Performance Rate", f"{bench['high_pct']:.1f}%",
                    f"In {category} category")
        col2.metric("Category Avg Views",    f"{bench['avg_views']/1e6:.2f}M",
                    "Per video average")
        col3.metric("Avg Days to Trending",  f"{bench['avg_days_trending']:.1f} days",
                    "After publishing")
