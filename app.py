import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import pandas as pd
import requests  # For API mode

# Init Spark (lightweight for inference)
@st.cache_resource
def load_model():
    spark = SparkSession.builder.appName("RecAPI").getOrCreate()
    model = ALSModel.load("svd_model")
    # Load all products for recs
    ratings = spark.read.parquet("ratings.parquet")
    products = ratings.select("productId").distinct().toPandas()
    return model, spark, products

model, spark, products = load_model()

st.title("Product Recommender API (SVD + Spark)")

# UI Mode
user_id = st.number_input("Enter User ID", min_value=1, value=12346)
if st.button("Get Recommendations"):
    # Generate recs
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recs = model.recommendForUserSubset(user_df, 10)
    recs_pd = recs.collect()[0].recommendations
    rec_list = [(r.productId, float(r.rating)) for r in recs_pd]
    st.write("Top-10 Product Recs (with predicted rating):")
    for pid, rating in rec_list:
        st.write(f"- Product {pid}: {rating:.2f}")

# API Mode (run with streamlit run app.py --server.headless true for API)
if 'api_mode' in st.query_params:
    user_id = int(st.query_params['user_id'][0])
    # ... (same rec logic, return JSON)
    import json
    response = {"user_id": user_id, "recommendations": rec_list}
    st.json(response)
    # For external calls: curl "http://localhost:8501/?api_mode=true&user_id=12346"