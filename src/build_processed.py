from __future__ import annotations
import os
import pandas as pd

try:
    from .features import weekly_features, make_label_next_week_slowdown
except ImportError:
    from features import weekly_features, make_label_next_week_slowdown


def safe_read_csv(path: str, columns: list[str]) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def build_processed() -> None:
    os.makedirs("data/processed", exist_ok=True)

    commits_path = "data/raw/commits.csv"
    prs_path = "data/raw/pull_requests.csv"
    reviews_path = "data/raw/reviews.csv"

    commits_cols = [
        "sha", "author_login", "author_name", "date", "message",
        "additions", "deletions", "total", "files_changed"
    ]
    prs_cols = [
        "pr_number", "author_login", "title", "created_at", "updated_at",
        "closed_at", "merged_at", "state", "comments", "review_comments",
        "commits", "additions", "deletions", "changed_files"
    ]
    reviews_cols = ["pr_number", "reviewer_login", "submitted_at", "state", "body"]

    if not os.path.exists(commits_path):
        print("Error: data/raw/commits.csv not found. Run build_dataset first.")
        return

    commits = safe_read_csv(commits_path, commits_cols)
    prs = safe_read_csv(prs_path, prs_cols)
    reviews = safe_read_csv(reviews_path, reviews_cols)

    feat = weekly_features(commits, prs, reviews)
    dataset = make_label_next_week_slowdown(feat, drop_ratio=0.6)

    dataset.to_csv("data/processed/dataset.csv", index=False)
    print("Saved data/processed/dataset.csv")


def main():
    build_processed()


if __name__ == "__main__":
    main()
