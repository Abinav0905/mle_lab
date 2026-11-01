import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

# the whole project folder (data/, utils/, scripts/, datamart/, model_bank/) is mounted here by docker-compose
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/app")


def run_script(script, extra_args=""):
    # {{ ds }} = the month this DAG run is for, e.g. 2024-01-01
    return "cd " + PROJECT_DIR + " && python3 scripts/" + script + " --snapshotdate {{ ds }} " + extra_args


def check_source(filename):
    # fails (and retries) if the source file is not there, so nothing downstream runs on missing data
    return "test -s " + PROJECT_DIR + "/data/" + filename + " && echo 'found data/" + filename + "'"


def decide_retrain(ds, **kwargs):
    """Model governance: retrain on schedule, or early when last month's monitoring flagged RED."""
    sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)
    import utils.model_utils as mu

    retrain, reason = mu.retrain_decision(ds)
    print("retrain:", retrain, "| reason:", reason)
    return "model_train" if retrain else "skip_training"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    "credit_default_ml_pipeline",
    default_args=default_args,
    description="monthly loan default pipeline: data -> feature & label store -> train -> predict -> monitor",
    schedule="0 0 1 * *",              # 00:00 on the 1st of every month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,                      # backfill every month from start_date to end_date
    max_active_runs=1,                 # one month at a time, in date order (a month needs the months before it)
    is_paused_upon_creation=False,     # start the backfill as soon as Airflow is up
    tags=["cs611"],
) as dag:

    # ------------------------------------------------------------------ label store
    check_source_lms = BashOperator(task_id="check_source_lms", bash_command=check_source("lms_loan_daily.csv"))
    bronze_lms = BashOperator(task_id="bronze_lms", bash_command=run_script("data_pipeline.py", "--layer bronze --source lms"))
    silver_loan_daily = BashOperator(task_id="silver_loan_daily", bash_command=run_script("data_pipeline.py", "--layer silver --source loan_daily"))
    gold_label_store = BashOperator(task_id="gold_label_store", bash_command=run_script("data_pipeline.py", "--layer gold --source label_store"))
    label_store_completed = EmptyOperator(task_id="label_store_completed")

    check_source_lms >> bronze_lms >> silver_loan_daily >> gold_label_store >> label_store_completed

    # ------------------------------------------------------------------ feature store
    gold_feature_store = BashOperator(task_id="gold_feature_store", bash_command=run_script("data_pipeline.py", "--layer gold --source feature_store"))
    feature_store_completed = EmptyOperator(task_id="feature_store_completed")

    source_files = {
        "attributes": "features_attributes.csv",
        "financials": "features_financials.csv",
        "clickstream": "feature_clickstream.csv",
    }
    for source, filename in source_files.items():
        check = BashOperator(task_id="check_source_" + source, bash_command=check_source(filename))
        bronze = BashOperator(task_id="bronze_" + source, bash_command=run_script("data_pipeline.py", "--layer bronze --source " + source))
        silver = BashOperator(task_id="silver_" + source, bash_command=run_script("data_pipeline.py", "--layer silver --source " + source))
        check >> bronze >> silver >> gold_feature_store

    silver_loan_daily >> gold_feature_store   # loan applications (month on book 0) come from the loan table
    gold_feature_store >> feature_store_completed

    # ------------------------------------------------------------------ model training (model governance)
    retrain_decision = BranchPythonOperator(task_id="retrain_decision", python_callable=decide_retrain)
    model_train = BashOperator(task_id="model_train", bash_command=run_script("model_train.py"))
    skip_training = EmptyOperator(task_id="skip_training")
    training_completed = EmptyOperator(task_id="training_completed", trigger_rule="none_failed_min_one_success")

    [label_store_completed, feature_store_completed] >> retrain_decision >> [model_train, skip_training] >> training_completed

    # ------------------------------------------------------------------ inference and monitoring
    # these scripts exit with code 99 when there is nothing to do yet (e.g. no model before 2024),
    # which Airflow shows as "skipped" instead of "failed"
    model_inference = BashOperator(task_id="model_inference", bash_command=run_script("model_inference.py"))
    model_monitoring = BashOperator(task_id="model_monitoring", bash_command=run_script("model_monitor.py"))
    monitoring_charts = BashOperator(task_id="monitoring_charts", bash_command=run_script("model_monitor_charts.py"))

    training_completed >> model_inference >> model_monitoring >> monitoring_charts
