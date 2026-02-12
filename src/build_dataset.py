from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.build_raw import build_raw
except ImportError:
    from build_raw import build_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    build_raw(args.owner, args.repo, args.days, token=args.token)
    print("Saved raw data to data/raw/")


if __name__ == "__main__":
    main()
