from __future__ import annotations
import re
import numpy as np
import pandas as pd

def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def add_time_features_commits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = _to_dt(df["date"])
    df["week"] = df["dt"].dt.to_period("W").astype(str)
    df["hour"] = df["dt"].dt.hour
    df["dow"] = df["dt"].dt.dayofweek  
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_after_hours"] = ((df["hour"] < 9) | (df["hour"] >= 19)).astype(int)
    df["churn"] = df["additions"].fillna(0) + df["deletions"].fillna(0)
    df["is_revert"] = df["message"].fillna("").str.lower().str.contains("revert").astype(int)
    df["is_bugfix"] = df["message"].fillna("").str.lower().str.contains(r"\b(fix|bug|hotfix)\b", regex=True).astype(int)
    df["developer"] = df["author_login"].fillna(df["author_name"]).fillna("unknown")
    return df

def add_time_features_prs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = _to_dt(df["created_at"])
    df["merged_at"] = _to_dt(df["merged_at"])
    df["closed_at"] = _to_dt(df["closed_at"])
    df["week"] = df["created_at"].dt.to_period("W").astype(str)
    df["developer"] = df["author_login"].fillna("unknown")
    end = df["merged_at"].fillna(df["closed_at"])
    df["cycle_time_hours"] = (end - df["created_at"]).dt.total_seconds() / 3600.0
    df["is_merged"] = df["merged_at"].notna().astype(int)
    return df

def weekly_features(commits: pd.DataFrame, prs: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    c = add_time_features_commits(commits)
    p = add_time_features_prs(prs)

    cagg = c.groupby(["developer", "week"]).agg(
        commits_count=("sha", "count"),
        lines_added=("additions", "sum"),
        lines_deleted=("deletions", "sum"),
        churn=("churn", "sum"),
        files_changed=("files_changed", "sum"),
        after_hours_commits=("is_after_hours", "sum"),
        weekend_commits=("is_weekend", "sum"),
        revert_commits=("is_revert", "sum"),
        bugfix_commits=("is_bugfix", "sum"),
    ).reset_index()

    cagg["after_hours_ratio"] = cagg["after_hours_commits"] / cagg["commits_count"].replace(0, np.nan)
    cagg["weekend_ratio"] = cagg["weekend_commits"] / cagg["commits_count"].replace(0, np.nan)
    cagg["bugfix_ratio"] = cagg["bugfix_commits"] / cagg["commits_count"].replace(0, np.nan)
    cagg = cagg.fillna(0)

    pagg = p.groupby(["developer", "week"]).agg(
        prs_opened=("pr_number", "count"),
        prs_merged=("is_merged", "sum"),
        avg_cycle_time_hours=("cycle_time_hours", "mean"),
        median_cycle_time_hours=("cycle_time_hours", "median"),
    ).reset_index().fillna(0)

    r = reviews.copy()
    if len(r) > 0:
        r["submitted_dt"] = _to_dt(r["submitted_at"])
        r["week"] = r["submitted_dt"].dt.to_period("W").astype(str)
        r["developer"] = r["reviewer_login"].fillna("unknown")
        ragg = r.groupby(["developer", "week"]).agg(
            reviews_done=("pr_number", "count")
        ).reset_index()
    else:
        ragg = pd.DataFrame(columns=["developer", "week", "reviews_done"])

    # merge
    feat = cagg.merge(pagg, on=["developer", "week"], how="outer").merge(ragg, on=["developer", "week"], how="outer")
    feat = feat.fillna(0)

    feat = feat.sort_values(["developer", "week"])
    feat["active_week"] = ((feat["commits_count"] + feat["prs_opened"] + feat["reviews_done"]) > 0).astype(int)
    feat["active_streak_weeks"] = feat.groupby("developer")["active_week"].transform(_streak)
    # trends
    feat["cycle_time_trend"] = feat.groupby("developer")["avg_cycle_time_hours"].diff().fillna(0)

    return feat

def _streak(s: pd.Series) -> pd.Series:
    out = []
    streak = 0
    for v in s.tolist():
        if v == 1:
            streak += 1
        else:
            streak = 0
        out.append(streak)
    return pd.Series(out, index=s.index)

def make_label_next_week_slowdown(feat: pd.DataFrame, drop_ratio: float = 0.6) -> pd.DataFrame:
    """
    Label=1 if next week commits drop by >= drop_ratio relative to this week.
    Example: drop_ratio=0.6 => next_week_commits <= 40% of this week.
    """
    df = feat.copy()
    df = df.sort_values(["developer", "week"])
    df["next_week_commits"] = df.groupby("developer")["commits_count"].shift(-1).fillna(0)

    df["slowdown_label"] = 0
    mask = df["commits_count"] >= 5
    df.loc[mask, "slowdown_label"] = (df.loc[mask, "next_week_commits"] <= (1 - drop_ratio) * df.loc[mask, "commits_count"]).astype(int)
    return df
