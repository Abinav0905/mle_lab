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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False) 
    print('loaded from:', filepath, 'row count:', df.count())

    # clean: remove stray underscores
    df = df.withColumn("Age", F.regexp_replace(col("Age"), "_", "").cast(IntegerType()))

    # clean: impossible ages 
    df = df.withColumn("Age", F.when((col("Age") >= 0) & (col("Age") <= 100), col("Age")))

    # clean: SSN format fix
    df = df.withColumn("SSN", F.when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN")))

    # clean: placeholder occupation
    df = df.withColumn("Occupation", F.when(col("Occupation") != "_______", col("Occupation")))

    # enforce schema
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table 
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)  
    print('loaded from:', filepath, 'row count:', df.count())

    # clean: remove stray underscores from number columns
    float_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit", "Outstanding_Debt",
                  "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]
    int_cols = ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
                "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]
    for c in float_cols:
        df = df.withColumn(c, F.regexp_replace(col(c), "_", "").cast(FloatType()))
    for c in int_cols:
        df = df.withColumn(c, F.regexp_replace(col(c), "_", "").cast(FloatType()).cast(IntegerType()))

    # clean: impossible values 
    valid_ranges = {
        "Num_Bank_Accounts": (0, 20),
        "Num_Credit_Card": (0, 20),
        "Interest_Rate": (0, 50),
        "Num_of_Loan": (0, 20),
        "Num_of_Delayed_Payment": (0, 50),
        "Num_Credit_Inquiries": (0, 50),
    }
    for c, (low, high) in valid_ranges.items():
        df = df.withColumn(c, F.when((col(c) >= low) & (col(c) <= high), col(c)))

    # clean: annual income should be ~12x monthly salary
    df = df.withColumn("Annual_Income", F.when(col("Annual_Income") <= 20 * col("Monthly_Inhand_Salary"), col("Annual_Income")))
    # clean: monthly loan repayments larger than monthly salary are data errors
    df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month") <= col("Monthly_Inhand_Salary"), col("Total_EMI_per_month")))
    # clean: 10000 is a placeholder value
    df = df.withColumn("Amount_invested_monthly", F.when(col("Amount_invested_monthly") != 10000, col("Amount_invested_monthly")))
    # clean: extreme negative balance is a placeholder
    df = df.withColumn("Monthly_Balance", F.when(col("Monthly_Balance") > -10000, col("Monthly_Balance")))

    # clean: placeholder categories
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix") != "_", col("Credit_Mix")))
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour") != "!@9#%8", col("Payment_Behaviour")))

    # augment: age in months
    df = df.withColumn("Credit_History_Age_Months",
                       (F.regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType()) * 12
                        + F.regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast(IntegerType())))

    # enforce schema
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table 
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # enforce schema: fe_1 ... fe_20 are whole numbers
    for i in range(1, 21):
        df = df.withColumn("fe_" + str(i), col("fe_" + str(i)).cast(IntegerType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table 
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df