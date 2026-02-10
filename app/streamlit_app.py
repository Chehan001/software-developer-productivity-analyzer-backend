from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[0]  
sys.path.append(str(PROJECT_ROOT))

try:
    from src.recommend import recommend_from_row
except Exception:
    st.error("Could not import recommendation engine. Ensure src/ is on PYTHONPATH.")
    st.stop()

st.set_page_config(page_title="Dev Productivity AI", layout="wide")


@st.cache_data
def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).fillna(0)
    return df


@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "slowdown_model.joblib"
    cols_path = PROJECT_ROOT / "models" / "feature_cols.joblib"
    if not model_path.exists() or not cols_path.exists():
        return None, None
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    return model, feature_cols


def line_chart(df, xcol, ycol, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[xcol], df[ycol], marker="o", linestyle="-")
    ax.set_title(title)
    ax.set_ylabel(ycol)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def main():
    st.title("AI Developer Productivity Analyzer (MVP)")

    df = load_data()
    if df.empty:
        st.warning("No data found. Run the pipeline first (build_dataset -> build_processed).")
        return

    if "developer" not in df.columns or "week" not in df.columns:
        st.error("Dataset missing required columns: 'developer' and/or 'week'.")
        return

    model, feature_cols = load_model()

    st.sidebar.header("Filters")
    devs = sorted(df["developer"].astype(str).unique().tolist())
    developer = st.sidebar.selectbox("Developer", devs)

    dev_df = df[df["developer"].astype(str) == str(developer)].sort_values("week").copy()

    if model is not None and feature_cols is not None and len(dev_df) > 0:
        X = dev_df.reindex(columns=feature_cols, fill_value=0)
        X = X.replace([float("inf"), float("-inf")], 0).fillna(0)
        try:
            probs = model.predict_proba(X)[:, 1]
            dev_df["slowdown_risk"] = probs
        except Exception as e:
            st.error(f"Model scoring failed: {e}")
            dev_df["slowdown_risk"] = 0.0
    else:
        dev_df["slowdown_risk"] = 0.0

    c1, c2, c3 = st.columns(3)
    latest = dev_df.iloc[-1] if len(dev_df) else None

    if latest is not None:
        c1.metric("Latest week", str(latest["week"]))
        if model is not None:
            c2.metric("Slowdown risk (0-1)", f'{float(latest["slowdown_risk"]):.2f}')
        else:
            c2.warning("Model not trained")
        c3.metric("Commits (latest week)", int(latest.get("commits_count", 0)))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Trends")
        if len(dev_df) > 0:
            if "commits_count" in dev_df.columns:
                line_chart(dev_df, "week", "commits_count", "Weekly Commits")
            if "avg_cycle_time_hours" in dev_df.columns:
                line_chart(dev_df, "week", "avg_cycle_time_hours", "Avg PR Cycle Time (hours)")
            if model is not None:
                line_chart(dev_df, "week", "slowdown_risk", "Predicted Slowdown Risk")
        else:
            st.info("No data for this developer.")

    with right:
        st.subheader("Burnout Signals (latest week)")
        if latest is not None:
            cols = st.columns(2)
            cols[0].write(f"**After-hours Ratio:** {round(float(latest.get('after_hours_ratio', 0)), 3)}")
            cols[0].write(f"**Weekend Ratio:** {round(float(latest.get('weekend_ratio', 0)), 3)}")
