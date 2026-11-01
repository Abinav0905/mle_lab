import pyspark


def get_spark(app_name="cs611_pipeline"):
    spark = pyspark.sql.SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")
    return spark
