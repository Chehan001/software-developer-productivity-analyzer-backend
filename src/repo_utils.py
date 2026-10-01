from __future__ import annotations

from urllib.parse import urlparse


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    value = (repo_url or "").strip()
    if not value:
        raise ValueError("Repository URL is required.")

    value = value.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]

    # Accept formats like:
    # https://github.com/owner/repo
    # http://github.com/owner/repo
    # github.com/owner/repo
    # git@github.com:owner/repo
    if value.startswith("git@github.com:"):
        value = value.replace("git@github.com:", "https://github.com/")

    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host not in {"github.com", "www.github.com"}:
        raise ValueError("Only GitHub repository URLs are supported.")

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Repository URL must be in the format https://github.com/owner/repo")

    owner, repo = parts[0], parts[1]
    if not owner or not repo:
        raise ValueError("Repository URL must include both owner and repo name.")

    return owner, repo
