from __future__ import annotations

import os
import pandas as pd
from datetime import datetime, timezone

try:
    from .github_client import GitHubClient
except ImportError:
    from github_client import GitHubClient


def iso_now_minus_days(days: int) -> str:
    dt = datetime.now(timezone.utc) - pd.Timedelta(days=days)
    return dt.isoformat()


def build_raw(owner: str, repo: str, days: int = 120, token: str = None) -> None:
    os.makedirs("data/raw", exist_ok=True)
    client = GitHubClient(token=token)

    since_iso = iso_now_minus_days(days)

    print(f"Fetching commits for {owner}/{repo} since {since_iso}...")
    commits_list = client.list_repo_commits(owner, repo, since_iso=since_iso, max_pages=2)

    commit_rows = []
    for item in commits_list:
        sha = item.get("sha")
        if not sha:
            continue

        commit = item.get("commit", {}) or {}
        author = commit.get("author", {}) or {}

        commit_rows.append({
            "sha": sha,
            "author_login": (item.get("author") or {}).get("login"),
            "author_name": author.get("name"),
            "date": author.get("date"),
            "message": commit.get("message"),
            "additions": 0,
            "deletions": 0,
            "total": 0,
            "files_changed": 0,
        })

    print(f"Fetching pull requests for {owner}/{repo}...")
    prs = client.list_pull_requests(owner, repo, state="all", max_pages=5)

    pr_rows = []
    review_rows = []

    for pr in prs:
        pr_number = pr.get("number")
        user = pr.get("user") or {}

        pr_rows.append({
            "pr_number": pr_number,
            "author_login": user.get("login"),
            "title": pr.get("title"),
            "created_at": pr.get("created_at"),
            "updated_at": pr.get("updated_at"),
            "closed_at": pr.get("closed_at"),
            "merged_at": pr.get("merged_at"),
            "state": pr.get("state"),
            "comments": pr.get("comments", 0),
            "review_comments": pr.get("review_comments", 0),
            "commits": None,
            "additions": None,
            "deletions": None,
            "changed_files": None,
        })
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

    pd.DataFrame(commit_rows, columns=commits_cols).to_csv("data/raw/commits.csv", index=False)
    pd.DataFrame(pr_rows, columns=prs_cols).to_csv("data/raw/pull_requests.csv", index=False)
    pd.DataFrame(review_rows, columns=reviews_cols).to_csv("data/raw/reviews.csv", index=False)

    print("Saved raw data to data/raw/")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    build_raw(args.owner, args.repo, args.days, token=args.token)


if __name__ == "__main__":
    main()
