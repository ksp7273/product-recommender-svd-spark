import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, count, sum as spark_sum
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField

# Initialize Spark
spark = SparkSession.builder.appName("DataPrep").config("spark.driver.memory", "4g").getOrCreate()

# Define schema explicitly
schema = StructType([
	StructField("Quantity", IntegerType(), True),
	StructField("UnitPrice", DoubleType(), True),
	StructField("CustomerID", IntegerType(), True),
	StructField("StockCode", IntegerType(), True),
	StructField("Description", StringType(), True),
	StructField("InvoiceNo", IntegerType(), True),
	StructField("InvoiceDate", StringType(), True),
	StructField("Country", StringType(), True)
])
df = spark.read.option("header", "true").schema(schema).csv("online_retail.csv")

# Debug: Print schema and sample data
print("Dataset Schema:")
df.printSchema()
print("Sample Data (5 rows):")
df.show(5, truncate=False)
# Check for required columns
required_columns = ["Quantity", "UnitPrice", "CustomerID", "StockCode"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
	print(f"Missing columns in CSV: {missing}")
	exit(1)

# Preprocess
df = df.withColumn("Quantity", col("Quantity").cast("int"))  # Check if 'Quantity' exists
df = df.withColumn("UnitPrice", col("UnitPrice").cast("double"))
df = df.filter(col("CustomerID").isNotNull() & (col("Quantity") > 0))

# Create ratings: proxy = min(Quantity * (UnitPrice / max_price) * 5, 5)
max_price = df.agg(spark_sum("UnitPrice")).collect()[0][0]
df = df.withColumn("Rating", when((col("Quantity") * col("UnitPrice") / lit(max_price) * 5) > 5, 5).otherwise(col("Quantity") * col("UnitPrice") / lit(max_price) * 5))
df = df.withColumn("Rating", col("Rating").cast(IntegerType()))

# Aggregate user-item ratings
ratings_df = df.groupBy("CustomerID", "StockCode").agg(spark_sum("Rating").alias("Rating")).withColumnRenamed("CustomerID", "userId").withColumnRenamed("StockCode", "productId")
# Filter out cold-start users/products
from pyspark.sql.functions import count as spark_count
# Count ratings per user and product, rename columns to avoid ambiguity
user_counts = ratings_df.groupBy("userId").agg(spark_count("productId").alias("user_num_rated"))
product_counts = ratings_df.groupBy("productId").agg(spark_count("userId").alias("product_num_rated"))
ratings_df = ratings_df.join(user_counts, on="userId").join(product_counts, on="productId")
ratings_df = ratings_df.filter((col("user_num_rated") >= 2) & (col("product_num_rated") >= 2))
ratings_df = ratings_df.drop("user_num_rated").drop("product_num_rated")
# Save as Parquet
ratings_df.write.mode("overwrite").parquet("ratings.parquet")
ratings_df.show(5)
spark.stop()
print("Dataset prepared")