import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, silver_loan_daily_directory, silver_attributes_directory,
                                silver_financials_directory, silver_clickstream_directory, gold_feature_store_directory, spark):
    date_suffix = snapshot_date_str.replace('-','_')

    # 1. find loans applied for this month 
    # because this is the moment the model will make its prediction
    df_loans = spark.read.parquet(silver_loan_daily_directory + "silver_loan_daily_" + date_suffix + ".parquet")
    df_loans = df_loans.filter(col("mob") == 0).select("loan_id", "Customer_ID", "snapshot_date")

    # 2. load this month's silver feature tables
    df_attr = spark.read.parquet(silver_attributes_directory + "silver_attributes_" + date_suffix + ".parquet")
    df_fin = spark.read.parquet(silver_financials_directory + "silver_financials_" + date_suffix + ".parquet")
    df_click = spark.read.parquet(silver_clickstream_directory + "silver_clickstream_" + date_suffix + ".parquet")

    # 3. attributes: drop personal info (privacy risk)
    df_attr = df_attr.select("Customer_ID", "Age", "Occupation")
    occupations = ["Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur", "Journalist", "Lawyer",
                   "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"]
    for occ in occupations:  
        df_attr = df_attr.withColumn("occ_" + occ, F.when(col("Occupation") == occ, 1).otherwise(0))
    df_attr = df_attr.drop("Occupation")

    # 4. financials: turn text categories into numbers
    df_fin = df_fin.withColumn("credit_mix_score",
                               F.when(col("Credit_Mix") == "Bad", 0).when(col("Credit_Mix") == "Standard", 1).when(col("Credit_Mix") == "Good", 2))
    df_fin = df_fin.withColumn("pays_min_amount_only",
                               F.when(col("Payment_of_Min_Amount") == "Yes", 1).when(col("Payment_of_Min_Amount") == "No", 0))
    df_fin = df_fin.withColumn("spend_level", 
                               F.when(col("Payment_Behaviour").startswith("Low"), 0).when(col("Payment_Behaviour").startswith("High"), 1))
    df_fin = df_fin.withColumn("payment_size",  # Small_value = 0, Medium = 1, Large = 2
                               F.when(col("Payment_Behaviour").contains("Small"), 0).when(col("Payment_Behaviour").contains("Medium"), 1)
                               .when(col("Payment_Behaviour").contains("Large"), 2))
    loan_types = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan",
                  "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"]
    for lt in loan_types:  # count how many of each loan type the customer already has
        df_fin = df_fin.withColumn("n_" + lt.replace(" ", "_").replace("-", "_"),
                                   F.coalesce(F.size(F.split(col("Type_of_Loan"), lt)) - 1, F.lit(0)))

    # 5. financials: ratios (affordability signals)
    df_fin = df_fin.withColumn("debt_to_income", col("Outstanding_Debt") / col("Annual_Income"))
    df_fin = df_fin.withColumn("emi_to_salary", col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
    df_fin = df_fin.drop("Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Type_of_Loan", "Credit_History_Age", "snapshot_date")

    # 6. clickstream: keep this month's fe_1..fe_20
    df_click = df_click.drop("snapshot_date")

    # 7. join everything onto the loans 
    df = df_loans.join(df_attr, on="Customer_ID", how="left") \
                 .join(df_fin, on="Customer_ID", how="left") \
                 .join(df_click, on="Customer_ID", how="left")

    # NOTE: missing values are left as null. Filling them (e.g. with the average) must be learned
    # from training data only, inside the ML pipeline to prevent data leak

    # save gold table 
    filepath = gold_feature_store_directory + "gold_feature_store_" + date_suffix + ".parquet"
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())

    return df