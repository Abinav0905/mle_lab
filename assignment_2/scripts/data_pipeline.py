import argparse
import os
import sys

# run from the project folder, so "data/", "datamart/" and "utils" are found
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold
from utils.spark_utils import get_spark

# to call this script: python scripts/data_pipeline.py --snapshotdate "2023-01-01" --layer bronze --source attributes

SOURCE_FILES = {
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
    "clickstream": "data/feature_clickstream.csv",
}

BRONZE_LMS_DIR = "datamart/bronze/lms/"
SILVER_LOAN_DAILY_DIR = "datamart/silver/loan_daily/"
GOLD_LABEL_STORE_DIR = "datamart/gold/label_store/"
GOLD_FEATURE_STORE_DIR = "datamart/gold/feature_store/"


def bronze_dir(source):
    return "datamart/bronze/" + source + "/"


def silver_dir(source):
    return "datamart/silver/" + source + "/"


def run(date_str, layer, source, spark):
    if layer == "bronze" and source == "lms":
        os.makedirs(BRONZE_LMS_DIR, exist_ok=True)
        bronze.process_bronze_table(date_str, BRONZE_LMS_DIR, spark)

    elif layer == "bronze" and source in SOURCE_FILES:
        os.makedirs(bronze_dir(source), exist_ok=True)
        bronze.process_bronze_feature_table(date_str, source, SOURCE_FILES[source], bronze_dir(source), spark)

    elif layer == "silver" and source == "loan_daily":
        os.makedirs(SILVER_LOAN_DAILY_DIR, exist_ok=True)
        silver.process_silver_table(date_str, BRONZE_LMS_DIR, SILVER_LOAN_DAILY_DIR, spark)

    elif layer == "silver" and source in SOURCE_FILES:
        silver_functions = {
            "attributes": silver.process_silver_attributes_table,
            "financials": silver.process_silver_financials_table,
            "clickstream": silver.process_silver_clickstream_table,
        }
        os.makedirs(silver_dir(source), exist_ok=True)
        silver_functions[source](date_str, bronze_dir(source), silver_dir(source), spark)

    elif layer == "gold" and source == "label_store":
        os.makedirs(GOLD_LABEL_STORE_DIR, exist_ok=True)
        gold.process_labels_gold_table(date_str, SILVER_LOAN_DAILY_DIR, GOLD_LABEL_STORE_DIR, spark, dpd=30, mob=6)

    elif layer == "gold" and source == "feature_store":
        os.makedirs(GOLD_FEATURE_STORE_DIR, exist_ok=True)
        gold.process_features_gold_table(date_str, SILVER_LOAN_DAILY_DIR, silver_dir("attributes"),
                                         silver_dir("financials"), silver_dir("clickstream"),
                                         GOLD_FEATURE_STORE_DIR, spark)

    else:
        raise ValueError("unknown layer / source combination: " + layer + " / " + source)


def main(snapshotdate, layer, source):
    print("\n\n---starting job---", layer, source, snapshotdate, "\n\n")
    spark = get_spark("data_pipeline_" + layer + "_" + source)
    try:
        run(snapshotdate, layer, source, spark)
    finally:
        spark.stop()
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run one step of the medallion data pipeline")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--layer", type=str, required=True, choices=["bronze", "silver", "gold"])
    parser.add_argument("--source", type=str, required=True,
                        help="lms | loan_daily | attributes | financials | clickstream | label_store | feature_store")
    args = parser.parse_args()
    main(args.snapshotdate, args.layer, args.source)
