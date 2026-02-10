from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURE_COLS = [
    "commits_count","lines_added","lines_deleted","churn","files_changed",
    "after_hours_ratio","weekend_ratio","revert_commits","bugfix_ratio",
    "prs_opened","prs_merged","avg_cycle_time_hours","median_cycle_time_hours",
    "reviews_done","active_streak_weeks","cycle_time_trend"
]

def train_model(dataset_csv: str = "data/processed/dataset.csv") -> None:
    if not os.path.exists(dataset_csv):
        print(f"Error: {dataset_csv} not found. Run build_processed first.")
        return

    df = pd.read_csv(dataset_csv)
    df = df.fillna(0)

    if len(df) < 5:
        print("Error: Not enough data to train. Need at least 5 developer-week rows.")
        return

    X = df[FEATURE_COLS]
    y = df["slowdown_label"].astype(int)

    # Simple check for class balance
    if y.nunique() < 2:
        print("Warning: Only one class found in labels. Model will be trivial.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    if y.nunique() > 1:
        proba = model.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, preds, digits=3))
        print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    else:
        print("Classification report not useful for single-class data.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/slowdown_model.joblib")
    joblib.dump(FEATURE_COLS, "models/feature_cols.joblib")
    print("Saved model to models/slowdown_model.joblib")

def main():
    train_model()

if __name__ == "__main__":
    main()
