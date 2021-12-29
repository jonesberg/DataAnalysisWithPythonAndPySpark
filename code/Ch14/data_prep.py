#!/usr/bin/env python3

# pylint: disable=missing-function-docstring

from typing import Optional

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Imputer, MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("Recipes ML model - Are you a dessert?")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

food = spark.read.csv(
    "./data/recipes/epi_r.csv", inferSchema=True, header=True
)


def sanitize_column_name(name):
    """Drops unwanted characters from the column name.

    We replace spaces, dashes and slashes with underscore,
    and only keep alphanumeric characters."""
    answer = name
    for i, j in ((" ", "_"), ("-", "_"), ("/", "_"), ("&", "and")):
        answer = answer.replace(i, j)
    return "".join(
        [
            char
            for char in answer
            if char.isalpha() or char.isdigit() or char == "_"
        ]
    )


food = food.toDF(*[sanitize_column_name(name) for name in food.columns])


# Keeping only the relevant values for `cakeweek` and `wasteless`.
# Check the exercises for a more robust approach to this.
food = food.where(
    (F.col("cakeweek").isin([0.0, 1.0]) | F.col("cakeweek").isNull())
    & (F.col("wasteless").isin([0.0, 1.0]) | F.col("wasteless").isNull())
)

IDENTIFIERS = ["title"]

CONTINUOUS_COLUMNS = [
    "rating",
    "calories",
    "protein",
    "fat",
    "sodium",
]

TARGET_COLUMN = ["dessert"]

BINARY_COLUMNS = [
    x
    for x in food.columns
    if x not in CONTINUOUS_COLUMNS
    and x not in TARGET_COLUMN
    and x not in IDENTIFIERS
]

food = food.dropna(
    how="all",
    subset=[x for x in food.columns if x not in IDENTIFIERS],
)

food = food.dropna(subset=TARGET_COLUMN)

@F.udf(T.BooleanType())
def is_a_number(value: Optional[str]) -> bool:
    if not value:
        return True
    try:
        _ = float(value)
    except ValueError:
        return False
    return True


for column in ["rating", "calories"]:
    food = food.where(is_a_number(F.col(column)))
    food = food.withColumn(column, F.col(column).cast(T.DoubleType()))

# TODO: REMOVE THIS
maximum = {
    "calories": 3203.0,
    "protein": 173.0,
    "fat": 207.0,
    "sodium": 5661.0,
}


inst_sum_of_binary_columns = [
    F.sum(F.col(x)).alias(x) for x in BINARY_COLUMNS
]

sum_of_binary_columns = (
    food.select(*inst_sum_of_binary_columns).head().asDict()
)

num_rows = food.count()
too_rare_features = [
    k
    for k, v in sum_of_binary_columns.items()
    if v < 10 or v > (num_rows - 10)
]

BINARY_COLUMNS = list(set(BINARY_COLUMNS) - set(too_rare_features))

food = food.withColumn(
    "protein_ratio", F.col("protein") * 4 / F.col("calories")
).withColumn("fat_ratio", F.col("fat") * 9 / F.col("calories"))

CONTINUOUS_COLUMNS += ["protein_ratio", "fat_ratio"]


from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features", labelCol="dessert", predictionCol="prediction"
)

from pyspark.ml import Pipeline
import pyspark.ml.feature as MF

imputer = MF.Imputer(  # <1>
    strategy="mean",
    inputCols=[
        "calories",
        "protein",
        "fat",
        "sodium",
        "protein_ratio",
        "fat_ratio",
    ],
    outputCols=[
        "calories_i",
        "protein_i",
        "fat_i",
        "sodium_i",
        "protein_ratio_i",
        "fat_ratio_i",
    ],
)

continuous_assembler = MF.VectorAssembler(
    inputCols=["rating", "calories_i", "protein_i", "fat_i", "sodium_i"],
    outputCol="continuous",
)

continuous_scaler = MF.MinMaxScaler(
    inputCol="continuous",
    outputCol="continuous_scaled",
)

preml_assembler = MF.VectorAssembler(
    inputCols=BINARY_COLUMNS
    + ["continuous_scaled"]
    + ["protein_ratio_i", "fat_ratio_i"],
    outputCol="features",
)
