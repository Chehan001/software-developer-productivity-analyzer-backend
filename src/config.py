import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_API = "https://api.github.com"

DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Only attach auth token looks real
if GITHUB_TOKEN and GITHUB_TOKEN.startswith(("ghp_", "github_pat_")):
    DEFAULT_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
