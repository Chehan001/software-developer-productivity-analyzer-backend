from __future__ import annotations

import time
import requests
from typing import Any, Dict, List, Optional

try:
    from .config import GITHUB_API, DEFAULT_HEADERS
except ImportError:
    from config import GITHUB_API, DEFAULT_HEADERS


class GitHubClient:
    def __init__(self, token: Optional[str] = None, sleep_s: float = 0.2):
        self.sleep_s = sleep_s
        self.token = token

    def _get(self, url: str, params: Optional[dict] = None) -> Any:
        headers = DEFAULT_HEADERS.copy()
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        r = requests.get(url, headers=headers, params=params, timeout=30)

      
        if r.status_code == 403 and "rate limit" in r.text.lower():
            time.sleep(5)
            r = requests.get(url, headers=DEFAULT_HEADERS, params=params, timeout=30)

        if not r.ok:
         
            raise RuntimeError(f"GitHub API error {r.status_code}: {r.text}")

        time.sleep(self.sleep_s)
        return r.json()

    def list_repo_commits(
        self,
        owner: str,
        repo: str,
        since_iso: str,
        per_page: int = 100,
        max_pages: int = 2,  
    ) -> List[Dict[str, Any]]:
        """
        Returns commit list items (not full commit details) since `since_iso`.
        """
        out: List[Dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
            data = self._get(url, params={"since": since_iso, "per_page": per_page, "page": page})
            if not data:
                break
            out.extend(data)
            page += 1
        return out

    def get_commit(self, owner: str, repo: str, sha: str) -> Dict[str, Any]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
        return self._get(url)

    def list_pull_requests(self, owner: str, repo: str, state: str = "all", max_pages: int = 5) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
            data = self._get(
                url,
                params={"state": state, "per_page": 100, "page": page, "sort": "updated", "direction": "desc"},
            )
            if not data:
                break
            out.extend(data)
            page += 1
        return out

    def list_reviews(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        return self._get(url, params={"per_page": 100, "page": 1})
