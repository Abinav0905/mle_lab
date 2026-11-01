import argparse
import json
import os
import pickle
import pprint
import sys
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import utils.model_utils as mu

# to call this script: python scripts/model_train.py --snapshotdate "2024-01-01"


def candidate_models():
    """3 candidate model types. Each is a sklearn Pipeline, so missing-value filling and scaling are
    learned from the training rows only and saved together with the model (no train-test contamination)."""
    return {
        "logistic_regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, solver="liblinear")),
            ]),
            {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                "model__penalty": ["l1", "l2"],
            },
        ),
        "random_forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=mu.RANDOM_STATE, n_jobs=-1)),
            ]),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [4, 6, 8, 12],
                "model__min_samples_leaf": [5, 10, 20, 50],
                "model__max_features": ["sqrt", 0.3, 0.5],
            },
        ),
        "xgboost": (
            Pipeline([
                # XGBoost handles missing values itself, so no imputer is needed
                ("model", XGBClassifier(eval_metric="logloss", tree_method="hist",
                                        random_state=mu.RANDOM_STATE, n_jobs=-1)),
            ]),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.6, 0.8],
                "model__min_child_weight": [1, 5, 10],
                "model__reg_lambda": [1, 5],
            },
        ),
    }


def gini(auc):
    return round(2 * auc - 1, 4)


def best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best = int(np.argmax(f1[:-1]))  # the last point has no threshold
    return float(thresholds[best])


def main(snapshotdate):
    print("\n\n---starting job---\n\n")
    train_date = mu.to_date(snapshotdate)
    model_version = mu.model_version_name(train_date)

    # --- 1. labels that have matured BEFORE the training date (no peeking into the future) ---
    labels = mu.read_partitions(mu.LABEL_STORE_DIR, "gold_label_store_", end=mu.add_months(train_date, -1))
    if labels.empty:
        raise ValueError("no matured labels before " + str(train_date))

    # the label is observed 6 months after the loan application; the features are dated at the application month
    labels["application_date"] = labels["snapshot_date"].apply(lambda d: mu.add_months(d, -mu.LABEL_MOB))
    latest_application = labels["application_date"].max()
    window_start = mu.add_months(latest_application, -(mu.TRAIN_WINDOW_MONTHS - 1))
    labels = labels[labels["application_date"] >= window_start]

    # --- 2. features at the application month, joined on loan_id ---
    features = mu.read_partitions(mu.FEATURE_STORE_DIR, "gold_feature_store_", start=window_start, end=latest_application)
    data = labels[["loan_id", "label", "application_date"]].merge(features, on="loan_id", how="inner")
    assert (data["snapshot_date"] == data["application_date"]).all(), "features must be dated at the application month"

    store_cols = [c for c in features.columns if c not in mu.ID_COLS]
    data[store_cols] = data[store_cols].astype(float)

    # --- feature availability check: a model can only use features that new applicants actually have ---
    current = mu.read_partition(mu.FEATURE_STORE_DIR, "gold_feature_store_", train_date)
    current_missing = current[store_cols].astype(float).isna().mean() if not current.empty else pd.Series(0.0, index=store_cols)
    train_missing = data[store_cols].isna().mean()
    excluded = {c: round(float(max(current_missing[c], train_missing[c])), 3) for c in store_cols
                if current_missing[c] > mu.MAX_MISSING_RATE or train_missing[c] > mu.MAX_MISSING_RATE}
    feature_cols = [c for c in store_cols if c not in excluded]
    if excluded:
        print("features left out (missing for more than", mu.MAX_MISSING_RATE, "of applicants):", sorted(excluded))

    # --- 3. split: most recent months = out-of-time (OOT) test; the rest = random 80/20 train/test ---
    application_months = sorted(data["application_date"].unique())
    oot_months = application_months[-mu.OOT_MONTHS:]
    oot = data[data["application_date"].isin(oot_months)]
    train_test = data[~data["application_date"].isin(oot_months)]

    X_train, X_test, y_train, y_test = train_test_split(
        train_test[feature_cols], train_test["label"],
        test_size=1 - mu.TRAIN_TEST_RATIO, random_state=mu.RANDOM_STATE, shuffle=True, stratify=train_test["label"])
    X_oot, y_oot = oot[feature_cols], oot["label"]

    config = {
        "model_version": model_version,
        "train_date": str(train_date),
        "train_test_months": [str(application_months[0]), str(application_months[-mu.OOT_MONTHS - 1])],
        "oot_months": [str(oot_months[0]), str(oot_months[-1])],
        "label_definition": labels["label_def"].iloc[0],
        "n_train": len(X_train), "n_test": len(X_test), "n_oot": len(X_oot),
        "default_rate_train": round(float(y_train.mean()), 4),
        "default_rate_test": round(float(y_test.mean()), 4),
        "default_rate_oot": round(float(y_oot.mean()), 4),
        "n_features": len(feature_cols),
        "n_features_excluded": len(excluded),
        "excluded_features": excluded,
    }
    pprint.pprint(config)

    # --- 4. tune each candidate with 3-fold cross validation on the TRAIN set only ---
    results = {}
    fitted = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=mu.RANDOM_STATE)
    for name, (pipeline, param_dist) in candidate_models().items():
        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, scoring="roc_auc", cv=cv,
                                    random_state=mu.RANDOM_STATE, n_jobs=1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        auc_oot = roc_auc_score(y_oot, model.predict_proba(X_oot)[:, 1])
        results[name] = {
            "cv_auc": round(float(search.best_score_), 4),
            "auc_train": round(auc_train, 4), "auc_test": round(auc_test, 4), "auc_oot": round(auc_oot, 4),
            "gini_train": gini(auc_train), "gini_test": gini(auc_test), "gini_oot": gini(auc_oot),
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        }
        fitted[name] = model
        print(name, results[name])

    # --- 5. select the best model on the TEST set (OOT stays untouched as the final check) ---
    best_name = max(results, key=lambda n: results[n]["auc_test"])
    best_model = fitted[best_name]
    print("\nselected model:", best_name, results[best_name])

    # decision threshold: the probability cut-off with the best F1 on the test set
    test_scores = best_model.predict_proba(X_test)[:, 1]
    threshold = best_f1_threshold(y_test, test_scores)

    # --- 6. reference distributions for stability monitoring (PSI) ---
    # scores: unseen data (test + OOT) is how the model behaves on new applicants
    reference_scores = np.concatenate([test_scores, best_model.predict_proba(X_oot)[:, 1]])
    score_bins = mu.make_bins(reference_scores)
    feature_reference = {}
    for c in feature_cols:
        edges = mu.make_bins(X_train[c])
        feature_reference[c] = {"bins": edges, "shares": mu.bin_shares(X_train[c], edges)}

    # which features drive the model (permutation importance on OOT)
    importance = permutation_importance(best_model, X_oot, y_oot, scoring="roc_auc", n_repeats=5,
                                        random_state=mu.RANDOM_STATE, n_jobs=1)
    top_features = pd.Series(importance.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)

    # --- 7. save the model artefact to the model bank ---
    artefact = {
        "model": best_model,
        "model_version": model_version,
        "model_type": best_name,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "config": config,
        "results": results,
        "selected_results": results[best_name],
        "reference_scores": {"bins": score_bins, "shares": mu.bin_shares(reference_scores, score_bins)},
        "reference_features": feature_reference,
        "top_features": {k: round(float(v), 5) for k, v in top_features.items()},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(mu.MODEL_BANK_DIR, exist_ok=True)
    artefact_path = os.path.join(mu.MODEL_BANK_DIR, model_version + ".pkl")
    with open(artefact_path, "wb") as f:
        pickle.dump(artefact, f)
    print("model saved to:", artefact_path)

    # human-readable summary next to the artefact
    summary = {k: v for k, v in artefact.items() if k not in ("model", "reference_features", "reference_scores")}
    with open(os.path.join(mu.MODEL_BANK_DIR, model_version + ".json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # model registry: one row per trained model version
    registry_path = os.path.join(mu.MODEL_BANK_DIR, "model_registry.csv")
    row = pd.DataFrame([{
        "model_version": model_version, "train_date": str(train_date), "model_type": best_name,
        "train_test_months": " to ".join(config["train_test_months"]), "oot_months": " to ".join(config["oot_months"]),
        "n_train": config["n_train"], "n_features": config["n_features"], "n_features_excluded": len(excluded),
        "auc_test": results[best_name]["auc_test"], "auc_oot": results[best_name]["auc_oot"],
        "gini_test": results[best_name]["gini_test"], "gini_oot": results[best_name]["gini_oot"],
        "threshold": round(threshold, 4), "artefact": artefact_path,
    }])
    if os.path.exists(registry_path):
        registry = pd.read_csv(registry_path)
        registry = registry[registry["model_version"] != model_version]  # re-runs replace the old row
        row = pd.concat([registry, row], ignore_index=True).sort_values("train_date")
    row.to_csv(registry_path, index=False)
    print("registry updated:", registry_path)

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train candidate models and save the best one to the model bank")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (training date)")
    args = parser.parse_args()
    main(args.snapshotdate)
