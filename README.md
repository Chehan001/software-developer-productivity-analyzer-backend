# AI Developer Productivity Analyzer (MVP)

This project pulls GitHub activity and builds weekly developer metrics, then trains a simple ML model to predict "next-week slowdown risk" and shows results in a Streamlit dashboard.

## Features
- GitHub API ingestion (commits, PRs, reviews)
- Weekly productivity metrics (commits, churn, PR cycle time, reviews)
- Burnout signals (after-hours ratio, weekend ratio, activity streak)
- ML prediction: next-week slowdown risk (Logistic Regression baseline)
- Streamlit dashboard + recommendations

## Setup
### 1) Create a GitHub token
Create a personal access token with `repo` read permissions for the repo you test.

### 2) Install & configure
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/macOS: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# edit .env and set GITHUB_TOKEN
```

### 3) Run pipeline
1. **Pull GitHub data**
   ```bash
   python -m src.build_dataset --owner <OWNER> --repo <REPO> --days 120
   ```
2. **Build processed dataset**
   ```bash
   python -m src.build_processed
   ```
3. **Train model**
   ```bash
   python -m src.train
   ```
4. **Start dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   ```






.\venv\Scripts\uvicorn src.api:app --reload