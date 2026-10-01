from __future__ import annotations

import json
import os
import sys
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Project root --> backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GITHUB_API  # noqa
from src.recommend import recommend_from_row  # noqa
from src.github_client import GitHubClient # noqa

app = FastAPI(title="Dev Productivity AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "slowdown_model.joblib"
COLS_PATH = PROJECT_ROOT / "models" / "feature_cols.joblib"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH).fillna(0)
    if "developer" not in df.columns or "week" not in df.columns:
        return pd.DataFrame()
    return df


def load_model():
    if not MODEL_PATH.exists() or not COLS_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(COLS_PATH)
    return model, feature_cols


def add_risk_scores(dev_df: pd.DataFrame, model, feature_cols) -> pd.DataFrame:
    dev_df = dev_df.copy()
    if model is None or feature_cols is None:
        dev_df["slowdown_risk"] = 0.0
        return dev_df

    X = dev_df.reindex(columns=feature_cols, fill_value=0)
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)
    dev_df["slowdown_risk"] = model.predict_proba(X)[:, 1]
    return dev_df


def repo_activity(owner: Optional[str], repo: Optional[str], token: Optional[str] = None, limit: int = 12, days: int = 30):
    if not owner or not repo:
        return {"commits": [], "pull_requests": []}

    client = GitHubClient(token=token)
    since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    commits = client.list_repo_commits(owner, repo, since_iso=since_iso, per_page=100, max_pages=1)
    recent_commits = []
    for item in commits[:limit]:
        commit = item.get("commit") or {}
        author = item.get("author") or {}
        recent_commits.append({
            "sha": item.get("sha"),
            "message": (commit.get("message") or "").split("\n")[0],
            "author_login": (author or {}).get("login"),
            "author_name": (commit.get("author") or {}).get("name"),
            "date": (commit.get("author") or {}).get("date"),
            "additions": 0,
            "deletions": 0,
        })

    pull_requests = client.list_pull_requests(owner, repo, state="all", max_pages=2)
    recent_prs = []
    for pr in pull_requests[:limit]:
        user = pr.get("user") or {}
        recent_prs.append({
            "pr_number": pr.get("number"),
            "title": pr.get("title"),
            "author_login": user.get("login"),
            "state": pr.get("state"),
            "created_at": pr.get("created_at"),
            "merged_at": pr.get("merged_at"),
        })

    return {"commits": recent_commits, "pull_requests": recent_prs}


def tail(s: str, n: int = 3000) -> str:
    return (s or "")[-n:]


def run_script(cmd: list[str]) -> Dict[str, Any]:
    """
    Run a subprocess from PROJECT_ROOT and capture output.
    Always prints stdout/stderr to backend terminal for debugging.
    """
    try:
        r = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        print("\n=== OK:", " ".join(cmd), "===")
        if r.stdout:
            print("--- STDOUT ---\n", r.stdout)
        if r.stderr:
            print("--- STDERR ---\n", r.stderr)

        return {
            "cmd": cmd,
            "returncode": r.returncode,
            "stdout_tail": tail(r.stdout),
            "stderr_tail": tail(r.stderr),
        }

    except subprocess.CalledProcessError as e:
        print("\n=== FAILED:", e.cmd, "===")
        print("EXIT:", e.returncode)
        print("--- STDOUT ---\n", e.stdout)
        print("--- STDERR ---\n", e.stderr)

        raise HTTPException(
            status_code=500,
            detail=(
                f"Refresh failed.\nCMD: {e.cmd}\n\n"
                f"EXIT: {e.returncode}\n\n"
                f"STDOUT:\n{e.stdout}\n\n"
                f"STDERR:\n{e.stderr}"
            ),
        )


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Dev Productivity AI API is running"}


@app.get("/api/developers")
async def get_developers():
    df = load_data()
    if df.empty:
        return []
    return sorted(df["developer"].astype(str).unique().tolist())


@app.get("/api/repository")
async def get_repository(owner: str, repo: str, token: Optional[str] = Query(default=None)):
    try:
        repository = GitHubClient(token=token).get_repository(owner, repo)
    except RuntimeError as exc:
        detail = str(exc)
        status_code = next((code for code in (401, 403, 404) if f" {code}:" in detail), 502)
        raise HTTPException(status_code=status_code, detail=detail)

    license_info = repository.get("license") or {}
    return {
        "full_name": repository.get("full_name"),
        "description": repository.get("description"),
        "html_url": repository.get("html_url"),
        "language": repository.get("language"),
        "stars": repository.get("stargazers_count", 0),
        "forks": repository.get("forks_count", 0),
        "open_issues": repository.get("open_issues_count", 0),
        "default_branch": repository.get("default_branch"),
        "license": license_info.get("spdx_id") or license_info.get("name"),
        "visibility": repository.get("visibility") or ("private" if repository.get("private") else "public"),
        "created_at": repository.get("created_at"),
        "pushed_at": repository.get("pushed_at"),
        "topics": repository.get("topics", []),
        "archived": repository.get("archived", False),
    }


def read_activity(path: Path, sort_column: str, limit: int) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return []
    if sort_column in frame.columns:
        frame = frame.sort_values(sort_column, ascending=False, na_position="last")
    return json.loads(frame.head(limit).to_json(orient="records", date_format="iso"))


@app.get("/api/activity")
async def get_activity(
    owner: Optional[str] = Query(default=None),
    repo: Optional[str] = Query(default=None),
    token: Optional[str] = Query(default=None),
    days: int = Query(default=30, ge=7, le=365),
    limit: int = Query(default=12, ge=1, le=50),
):
    if owner or repo:
        if not owner or not repo:
            raise HTTPException(status_code=400, detail="Both owner and repo are required")
        try:
            return repo_activity(owner, repo, token=token, limit=limit, days=days)
        except RuntimeError as exc:
            detail = str(exc)
            status_code = next((code for code in (401, 403, 404) if f" {code}:" in detail), 502)
            raise HTTPException(status_code=status_code, detail=detail)

    raw_path = PROJECT_ROOT / "data" / "raw"
    return {
        "commits": read_activity(raw_path / "commits.csv", "date", limit),
        "pull_requests": read_activity(raw_path / "pull_requests.csv", "created_at", limit),
    }


@app.get("/api/developer/{name}")
async def get_developer_stats(name: str):
    df = load_data()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found at {DATA_PATH}")

    dev_df = df[df["developer"].astype(str) == str(name)].sort_values("week")
    if dev_df.empty:
        raise HTTPException(status_code=404, detail=f"Developer '{name}' not found")

    model, feature_cols = load_model()
    try:
        dev_df = add_risk_scores(dev_df, model, feature_cols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk scoring failed: {e}")

    records = dev_df.to_dict(orient="records")
    latest = records[-1] if records else {}

    try:
        recommendations = recommend_from_row(dev_df.iloc[-1]) if len(dev_df) else []
    except Exception as e:
        recommendations = [f"Recommendation error: {e}"]

    return {
        "developer": name,
        "history": records,
        "latest": latest,
        "recommendations": recommendations,
        "model_trained": model is not None,
    }


@app.get("/api/github/repos")
async def search_repos(token: str, query: str = ""):
    """
    Search/List user repositories using a github token.
    """
    if not token or not token.strip():
        raise HTTPException(status_code=400, detail="Token required")
    
    try:
        client = GitHubClient(token=token)
        #User All Git Repositories 
        url = "https://api.github.com/user/repos"
        repos = client._get(url, params={"sort": "updated", "per_page": 50})
        
        if query:
            repos = [r for r in repos if query.lower() in r.get("full_name", "").lower()]
            
        return [
            {
                "full_name": r.get("full_name"),
                "name": r.get("name"),
                "owner": r.get("owner", {}).get("login"),
            }
            for r in repos
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh")
async def refresh_data(
    owner: Optional[str] = Query(default=None),
    repo: Optional[str] = Query(default=None),
    days: int = Query(default=120),
    token: Optional[str] = Query(default=None),
):
    owner = (owner or os.getenv("GITHUB_OWNER") or "").strip()
    repo = (repo or os.getenv("GITHUB_REPO") or "").strip()

    if not owner or not repo:
        raise HTTPException(
            status_code=400,
            detail="Missing owner/repo. Use /api/refresh?owner=ORG&repo=REPO&days=10",
        )

    build_dataset_path = PROJECT_ROOT / "src" / "build_dataset.py"
    build_processed_path = PROJECT_ROOT / "src" / "build_processed.py"
    train_model_path = PROJECT_ROOT / "src" / "train.py"

    if not build_dataset_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {build_dataset_path}")
    if not build_processed_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing script: {build_processed_path}")

    cmd1 = [
        sys.executable,
        str(build_dataset_path),
        "--owner", owner,
        "--repo", repo,
        "--days", str(days),
    ]
    if token:
        cmd1.extend(["--token", token])

    out1 = run_script(cmd1)

    out2 = run_script([
        sys.executable,
        str(build_processed_path),
    ])

    out3 = None
    if train_model_path.exists():
        out3 = run_script([
            sys.executable,
            str(train_model_path),
        ])

    return {
        "status": "success",
        "message": f"Refreshed for {owner}/{repo} (days={days})",
        "dataset_path": str(DATA_PATH),
        "build_dataset": out1,
        "build_processed": out2,
        "train_model": out3,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
