import os
import glob
import re
import pickle
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pipeline settings (one place to change them)
# ---------------------------------------------------------------------------

# datamart locations (relative to the project folder)
FEATURE_STORE_DIR = "datamart/gold/feature_store/"
LABEL_STORE_DIR = "datamart/gold/label_store/"
PREDICTIONS_DIR = "datamart/gold/model_predictions/"
MONITORING_DIR = "datamart/gold/model_monitoring/"
FEATURE_DRIFT_DIR = "datamart/gold/model_feature_drift/"
CHARTS_DIR = "datamart/gold/model_monitoring_charts/"
MODEL_BANK_DIR = "model_bank/"

# label definition (must match the label store): default = 30+ days past due at month 6 on book
LABEL_MOB = 6

# model governance (see retrain_decision below)
FIRST_TRAIN_DATE = "2024-01-01"   # first model: enough matured labels (6 application months) exist by then
RETRAIN_EVERY_MONTHS = 12         # scheduled refresh: once a year
DRIFT_RETRAIN = True              # early refresh: when last month's monitoring flagged RED for the production model
MAX_MISSING_RATE = 0.5            # at training time, leave out features missing for more than 50% of current applicants

# training data window (in loan application months)
TRAIN_WINDOW_MONTHS = 12   # use at most the 12 most recent application months that have a matured label
OOT_MONTHS = 2             # the 2 most recent of those months are kept aside as the out-of-time test
TRAIN_TEST_RATIO = 0.8     # the rest is split 80% train / 20% test
RANDOM_STATE = 88

# monitoring thresholds
PSI_AMBER = 0.10           # PSI 0.10 - 0.25: distribution is shifting, watch closely
PSI_RED = 0.25             # PSI > 0.25: distribution has shifted, action needed
GINI_DROP_AMBER = 0.05     # Gini 0.05 - 0.10 below the model's out-of-time Gini: watch closely
GINI_DROP_RED = 0.10       # Gini > 0.10 below the model's out-of-time Gini: action needed

ID_COLS = ["loan_id", "Customer_ID", "snapshot_date"]

# BashOperator marks a task as "skipped" (not failed) when the script exits with this code
SKIP_EXIT_CODE = 99


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def to_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def add_months(d, months):
    return to_date(d) + relativedelta(months=months)


def months_between(start, end):
    start, end = to_date(start), to_date(end)
    return (end.year - start.year) * 12 + (end.month - start.month)


def date_suffix(d):
    return to_date(d).strftime("%Y_%m_%d")


def partition_date(filepath):
    # every partition ends with _YYYY_MM_DD.parquet, e.g. gold_label_store_2023_07_01.parquet
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})\.(parquet|pkl)$", os.path.basename(filepath.rstrip("/")))
    if match is None:
        return None
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


# ---------------------------------------------------------------------------
# Reading gold tables (Spark writes each partition as a folder of parquet files)
# ---------------------------------------------------------------------------

def read_partitions(directory, prefix, start=None, end=None):
    """Read every partition <directory><prefix>YYYY_MM_DD.parquet with start <= date <= end."""
    frames = []
    for path in sorted(glob.glob(os.path.join(directory, prefix + "*.parquet"))):
        d = partition_date(path)
        if d is None:
            continue
        if start is not None and d < to_date(start):
            continue
        if end is not None and d > to_date(end):
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def read_partition(directory, prefix, d):
    return read_partitions(directory, prefix, start=d, end=d)


def write_parquet(df, directory, prefix, d):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, prefix + date_suffix(d) + ".parquet")
    df.to_parquet(filepath, index=False)
    print("saved to:", filepath, "row count:", len(df))
    return filepath


# ---------------------------------------------------------------------------
# Model bank
# ---------------------------------------------------------------------------

def model_version_name(train_date):
    return "credit_model_" + date_suffix(train_date)


def list_models():
    """All model artefacts in the model bank, oldest first: [(train_date, filepath), ...]"""
    models = []
    for path in glob.glob(os.path.join(MODEL_BANK_DIR, "credit_model_*.pkl")):
        d = partition_date(path)
        if d is not None:
            models.append((d, path))
    return sorted(models)


def latest_model_path(as_of_date):
    """The production model on a given date = the newest model trained on or before that date."""
    eligible = [path for d, path in list_models() if d <= to_date(as_of_date)]
    return eligible[-1] if eligible else None


def load_model(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def retrain_decision(as_of_date):
    """Model governance rule, used by the Airflow DAG every month. Returns (retrain: bool, reason: str).

    1. no model before FIRST_TRAIN_DATE (not enough matured labels yet)
    2. scheduled refresh on FIRST_TRAIN_DATE and every RETRAIN_EVERY_MONTHS after it
    3. early refresh if last month's monitoring flagged action_required for the model that was in production

    Only models trained BEFORE this month count, so re-running (backfilling) the pipeline always gives the same result.
    """
    as_of_date = to_date(as_of_date)
    first = to_date(FIRST_TRAIN_DATE)
    if as_of_date < first:
        return False, "before first training date " + str(first)

    months = months_between(first, as_of_date)
    if months % RETRAIN_EVERY_MONTHS == 0:
        return True, "scheduled refresh (every " + str(RETRAIN_EVERY_MONTHS) + " months)"

    previous = [path for d, path in list_models() if d < as_of_date]
    if not previous:
        return True, "no production model yet"

    if DRIFT_RETRAIN:
        production_version = os.path.basename(previous[-1])[:-4]
        last_month = read_partition(MONITORING_DIR, "model_monitoring_", add_months(as_of_date, -1))
        if not last_month.empty:
            last = last_month.iloc[0]
            if bool(last["action_required"]) and last["model_version"] == production_version:
                return True, "early refresh: monitoring flagged RED for " + production_version + " last month"

    return False, "no refresh needed"


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def make_bins(values, n_bins=10):
    """Bin edges = the deciles of the reference data (duplicates removed, e.g. for 0/1 columns)."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return edges.tolist()


def bin_shares(values, edges):
    """Share of rows in each bin, plus one extra bin for missing values."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    missing = np.isnan(values)
    counts = np.bincount(np.digitize(values[~missing], edges, right=True), minlength=len(edges) + 1)
    counts = np.append(counts, missing.sum()).astype(float)
    total = counts.sum()
    return (counts / total).tolist() if total > 0 else counts.tolist()


def psi(reference_shares, current_shares, eps=1e-4):
    ref = np.clip(np.asarray(reference_shares, dtype=float), eps, None)
    cur = np.clip(np.asarray(current_shares, dtype=float), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_status(value):
    if value is None or pd.isna(value):
        return None
    if value > PSI_RED:
        return "RED"
    if value > PSI_AMBER:
        return "AMBER"
    return "GREEN"


def gini_status(current_gini, reference_gini):
    if current_gini is None or pd.isna(current_gini):
        return None
    drop = reference_gini - current_gini
    if drop > GINI_DROP_RED:
        return "RED"
    if drop > GINI_DROP_AMBER:
        return "AMBER"
    return "GREEN"
