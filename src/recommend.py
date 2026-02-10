from __future__ import annotations
import pandas as pd

def recommend_from_row(row: pd.Series) -> list[str]:
    recs = []

    if row.get("after_hours_ratio", 0) > 0.35:
        recs.append("High after-hours activity. Consider limiting late-night work and spreading tasks across the week.")

    if row.get("weekend_ratio", 0) > 0.2:
        recs.append("Frequent weekend activity detected. Suggest setting boundaries or rotating on-call/support duties.")

    if row.get("active_streak_weeks", 0) >= 6:
        recs.append("Long continuous streak of active weeks. Suggest scheduling lighter sprint or taking planned break.")

    if row.get("avg_cycle_time_hours", 0) > 72:
        recs.append("PR cycle time is high. Suggest smaller PRs, clear review ownership, and a review SLA (e.g., 24h).")

    if row.get("cycle_time_trend", 0) > 8:
        recs.append("PR cycle time is increasing week-over-week. Investigate review bottlenecks or unclear requirements.")

    if row.get("churn", 0) > 4000:
        recs.append("Large code churn. Suggest breaking work into smaller deliverables to reduce risk and review time.")

    if row.get("revert_commits", 0) >= 1:
        recs.append("Revert activity detected. Suggest stronger testing, smaller PRs, or feature flags.")

    if not recs:
        recs.append("No major risk signals. Keep stable pace and continue improving PR review hygiene.")

    return recs
