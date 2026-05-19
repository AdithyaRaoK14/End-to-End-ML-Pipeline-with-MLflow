# src/retrain.py
from train import train_and_log
import schedule
import time
import mlflow
from mlflow.tracking import MlflowClient
import os
import sys

sys.path.append(os.path.dirname(__file__))

mlflow.set_tracking_uri("http://localhost:5000")

DATA_PATH = os.path.join(os.path.dirname(
    __file__), "..", "data", "creditcard.csv")

BEST_CONFIG = {
    "run_name":      "scheduled-retrain",
    "model":         "RandomForest",
    "sampling":      "smote",
    "n_estimators":  100,
    "max_depth":     10,
    "class_weight":  None,
}


def retrain_job():
    print(f"\n{'='*55}")
    print(f"  SCHEDULED RETRAIN — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")

    client = MlflowClient()

    # ── Get current Production model metrics ──────────────────────────────
    try:
        current = client.get_latest_versions(
            "fraud-detector", stages=["Production"])[0]
        current_run = client.get_run(current.run_id)
        current_recall = float(current_run.data.metrics["recall_fraud"])
        current_auc = float(current_run.data.metrics["roc_auc"])
        current_version = current.version
        print(f"  Current Production: v{current_version}")
        print(f"  Recall: {current_recall:.4f} | ROC-AUC: {current_auc:.4f}")
    except Exception as e:
        print(f"  No Production model found: {e}")
        print("  Training from scratch...")
        current_recall = 0.0
        current_auc = 0.0
        current_version = None

    # ── Retrain ───────────────────────────────────────────────────────────
    print("\n  Retraining...")
    new_metrics = train_and_log(BEST_CONFIG, data_path=DATA_PATH)

    # ── Compare and Promote ───────────────────────────────────────────────
    # Promote only if recall improves — missing fraud is more costly
    if new_metrics["recall_fraud"] > current_recall:
        print(f"\n  ✅ New model better!")
        print(
            f"  Recall: {current_recall:.4f} → {new_metrics['recall_fraud']:.4f}")

        # Get the version that was just registered
        # REPLACE the promotion block in retrain_job() with this:
        try:
            latest = client.get_latest_versions("fraud-detector")[0]
            client.set_registered_model_alias(
                "fraud-detector", "production", str(latest.version)
            )
            print(f"  Promoted v{latest.version} to production alias")
        except Exception as e:
            print(f"  Could not promote: {e}")
    else:
        print(f"\n  Current model still best.")
        print(
            f"  Old recall: {current_recall:.4f} | New recall: {new_metrics['recall_fraud']:.4f}")
        print(f"  Keeping v{current_version} in Production.")

    print(f"\n  Next retrain in 24 hours.")


# ── Run once immediately on start, then schedule ──────────────────────────────
if __name__ == "__main__":
    print("Retrain scheduler started.")
    print("Running initial retrain now, then every 24 hours...")

    retrain_job()   # run immediately so you can test it

    schedule.every(24).hours.do(retrain_job)

    while True:
        schedule.run_pending()
        time.sleep(60)
