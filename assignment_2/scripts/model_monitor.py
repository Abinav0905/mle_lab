import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import utils.model_utils as mu

# to call this script: python scripts/model_monitor.py --snapshotdate "2024-07-01"
#
# Each month this job checks two things:
#   STABILITY   - are this month's applicants / scores still like the ones the model was built on? (PSI, no labels needed)
#   PERFORMANCE - for the loans scored 6 months ago, whose outcome is now known, how good were the predictions? (AUC / Gini)


def main(snapshotdate):
    print("\n\n---starting job---\n\n")
    snapshot_date = mu.to_date(snapshotdate)

    predictions = mu.read_partition(mu.PREDICTIONS_DIR, "model_predictions_", snapshot_date)
    if predictions.empty:
        print("no predictions for", snapshot_date, "- nothing to monitor, skipping")
        sys.exit(mu.SKIP_EXIT_CODE)

    model_version = predictions["model_version"].iloc[0]
    artefact = mu.load_model(os.path.join(mu.MODEL_BANK_DIR, model_version + ".pkl"))
    reference_gini = artefact["selected_results"]["gini_oot"]

    row = {
        "snapshot_date": snapshot_date,
        "model_version": model_version,
        "model_type": artefact["model_type"],
        "n_scored": len(predictions),
        "mean_predicted_probability": float(predictions["default_probability"].mean()),
        "predicted_default_rate": float(predictions["predicted_default"].mean()),
    }

    # --- 1. STABILITY: score distribution vs the model's reference (PSI) ---
    ref = artefact["reference_scores"]
    row["score_psi"] = mu.psi(ref["shares"], mu.bin_shares(predictions["default_probability"], ref["bins"]))

    # feature drift: PSI and missing rate of every model feature vs the training data
    # (PSI has a separate "missing" bin, so a broken data feed shows up as a very high PSI)
    features = mu.read_partition(mu.FEATURE_STORE_DIR, "gold_feature_store_", snapshot_date)
    drift_rows = []
    for c, r in artefact["reference_features"].items():
        current_values = features[c].astype(float) if c in features.columns else pd.Series(np.nan, index=features.index)
        drift_rows.append({"snapshot_date": snapshot_date, "model_version": model_version, "feature": c,
                           "psi": mu.psi(r["shares"], mu.bin_shares(current_values, r["bins"])),
                           "missing_rate": float(current_values.isna().mean()),
                           "reference_missing_rate": float(r["shares"][-1]),
                           "is_top_feature": c in artefact["top_features"]})
    drift = pd.DataFrame(drift_rows)
    drift["status"] = drift["psi"].apply(mu.psi_status)
    top = drift.sort_values("psi", ascending=False).iloc[0]
    row["max_feature_psi"] = float(top["psi"])
    row["max_feature_psi_name"] = top["feature"]
    row["n_features_psi_red"] = int((drift["status"] == "RED").sum())
    row["n_features_mostly_missing"] = int((drift["missing_rate"] > mu.MAX_MISSING_RATE).sum())
    row["stability_status"] = mu.psi_status(max(row["score_psi"], row["max_feature_psi"]))

    # --- 2. PERFORMANCE: loans scored LABEL_MOB months ago now have a known outcome ---
    cohort_date = mu.add_months(snapshot_date, -mu.LABEL_MOB)
    cohort_predictions = mu.read_partition(mu.PREDICTIONS_DIR, "model_predictions_", cohort_date)
    labels = mu.read_partition(mu.LABEL_STORE_DIR, "gold_label_store_", snapshot_date)
    row["performance_cohort_date"] = cohort_date
    row["performance_model_version"] = None
    for metric in ["n_labelled", "actual_default_rate", "cohort_mean_predicted_probability",
                   "auc", "gini", "accuracy", "precision", "recall", "f1", "reference_gini_oot", "gini_drop"]:
        row[metric] = np.nan

    if not cohort_predictions.empty and not labels.empty:
        scored = cohort_predictions.merge(labels[["loan_id", "label"]], on="loan_id", how="inner")
        if len(scored) > 0 and scored["label"].nunique() == 2:
            cohort_version = scored["model_version"].iloc[0]
            cohort_artefact = artefact if cohort_version == model_version else \
                mu.load_model(os.path.join(mu.MODEL_BANK_DIR, cohort_version + ".pkl"))
            cohort_reference_gini = cohort_artefact["selected_results"]["gini_oot"]
            auc = roc_auc_score(scored["label"], scored["default_probability"])
            row.update({
                "performance_model_version": cohort_version,
                "n_labelled": len(scored),
                "actual_default_rate": float(scored["label"].mean()),
                "cohort_mean_predicted_probability": float(scored["default_probability"].mean()),
                "auc": auc,
                "gini": 2 * auc - 1,
                "accuracy": accuracy_score(scored["label"], scored["predicted_default"]),
                "precision": precision_score(scored["label"], scored["predicted_default"], zero_division=0),
                "recall": recall_score(scored["label"], scored["predicted_default"], zero_division=0),
                "f1": f1_score(scored["label"], scored["predicted_default"], zero_division=0),
                "reference_gini_oot": cohort_reference_gini,
                "gini_drop": cohort_reference_gini - (2 * auc - 1),
            })
            reference_gini = cohort_reference_gini
    else:
        print("no matured outcomes yet for the", cohort_date, "cohort - performance not available this month")

    row["performance_status"] = mu.gini_status(row["gini"], reference_gini)

    # action is only needed for the model that is in production NOW
    # (a RED for an older, already-replaced model is recorded, but does not trigger a refresh)
    performance_red_for_production = row["performance_status"] == "RED" and row["performance_model_version"] == model_version
    row["action_required"] = bool(row["stability_status"] == "RED" or performance_red_for_production)

    # --- 3. save both monitoring gold tables ---
    summary = pd.DataFrame([row])
    mu.write_parquet(summary, mu.MONITORING_DIR, "model_monitoring_", snapshot_date)
    mu.write_parquet(drift, mu.FEATURE_DRIFT_DIR, "model_feature_drift_", snapshot_date)
    print(summary.T.to_string())

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monitor model performance and stability for one month")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
