import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import utils.model_utils as mu

# to call this script: python scripts/model_inference.py --snapshotdate "2024-01-01"


def main(snapshotdate):
    print("\n\n---starting job---\n\n")
    snapshot_date = mu.to_date(snapshotdate)

    # --- 1. retrieve the production model: newest model trained on or before this month ---
    model_path = mu.latest_model_path(snapshot_date)
    if model_path is None:
        print("no model in the model bank trained on or before", snapshot_date, "- skipping inference")
        sys.exit(mu.SKIP_EXIT_CODE)
    artefact = mu.load_model(model_path)
    print("model loaded:", model_path, "(" + artefact["model_type"] + ")")

    # --- 2. this month's loan applications from the feature store ---
    features = mu.read_partition(mu.FEATURE_STORE_DIR, "gold_feature_store_", snapshot_date)
    if features.empty:
        print("no loan applications in the feature store for", snapshot_date, "- skipping inference")
        sys.exit(mu.SKIP_EXIT_CODE)

    missing_cols = [c for c in artefact["feature_cols"] if c not in features.columns]
    if missing_cols:
        raise ValueError("feature store is missing columns the model needs: " + str(missing_cols))
    X = features[artefact["feature_cols"]].astype(float)

    # --- 3. predict ---
    scores = artefact["model"].predict_proba(X)[:, 1]
    predictions = features[mu.ID_COLS].copy()
    predictions["model_version"] = artefact["model_version"]
    predictions["model_type"] = artefact["model_type"]
    predictions["default_probability"] = scores
    predictions["threshold"] = artefact["threshold"]
    predictions["predicted_default"] = (scores >= artefact["threshold"]).astype(int)

    # --- 4. save as a gold table (one partition per month) ---
    mu.write_parquet(predictions, mu.PREDICTIONS_DIR, "model_predictions_", snapshot_date)
    print("applications scored:", len(predictions), "| predicted defaults:", int(predictions["predicted_default"].sum()))

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score this month's loan applications with the production model")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
