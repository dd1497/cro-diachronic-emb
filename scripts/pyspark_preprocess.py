from ast import literal_eval

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession


@F.udf(T.ArrayType(T.ArrayType(T.StringType())))
def string2list(s):
    return literal_eval(s)


@F.udf(T.ArrayType(T.ArrayType(T.StringType())))
def join_arrays(arr1, arr2):
    """
    Joins two arrays of arrays into a single array of arrays.
    Each inner array is joined with the corresponding inner array from the other array.
    """
    if arr1 is None or arr2 is None:
        return None

    if len(arr1) != len(arr2):
        return None

    arr_new = []

    for arr1_, arr2_ in zip(arr1, arr2):
        arr_new.append(
            [
                e1.lower().strip() + "_" + e2.lower().strip()
                for e1, e2 in zip(arr1_, arr2_)
            ]
        )

    return arr_new


@F.udf(T.StringType())
def extract_year(date_published):
    return str(date_published.split()[0].split("-")[0])


@F.udf(T.StringType())
def extract_month(date_published):
    return str(date_published.split()[0].split("-")[1])


@F.udf(T.StringType())
def extract_day(date_published):
    return str(date_published.split()[0].split("-")[2])


def start_spark(config=None, appName="retriever", n_threads=48):
    spark = (
        SparkSession.builder.master(f"local[{n_threads}]")
        .appName(appName)
        .config("spark.driver.memory", "70g")
        .config("spark.executor.memory", "40g")
        .config("spark.local.dir", "/tmp")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "16g")
    )
    if config is not None:
        for k, v in config:
            spark = spark.config(k, v)
    return spark.getOrCreate()


# change input and output file names to match your parquet data
input_file = "retriever-diachronic-emb/data/raw/diachronic_wave_4.parquet"
output_file = (
    "retriever-diachronic-emb/data/processed/diachronic_wave_4_processed.parquet"
)

spark = start_spark()

df = spark.read.parquet(input_file)

df = df.select(
    string2list("tokens_lemmas").alias("lemmas"),
    string2list("pos_tags").alias("tags"),
    "portal",
    "date_published",
)


df = df.withColumn("lemma_pos", join_arrays("lemmas", "tags"))

df = df.withColumn("year", extract_year("date_published"))

df = df.withColumn("month", extract_month("date_published"))

df = df.withColumn("day", extract_day("date_published"))

df = df.drop("lemmas", "tags", "date_published")

df.coalesce(1).write.parquet(output_file)
