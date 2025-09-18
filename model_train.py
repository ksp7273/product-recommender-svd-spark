from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS  # ALS uses SVD under the hood in MLlib
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
from sklearn.metrics import precision_score  # For Precision@K (custom impl)

# Init Spark
spark = SparkSession.builder.appName("SVDRecSys").config("spark.driver.memory", "4g").getOrCreate()

# Load data
ratings = spark.read.parquet("ratings.parquet")
(row_count, train_count) = (ratings.count(), 0)
if row_count == 0:
    print("ERROR: ratings.parquet is empty. Check data_prep.py output.")
    spark.stop()
    exit(1)
(training, test) = ratings.randomSplit([0.8, 0.2], seed=42)
train_count = training.count()
print(f"Total ratings: {row_count}, Training set: {train_count}")
if train_count == 0:
    print("ERROR: Training set is empty. Check filtering and aggregation in data_prep.py.")
    spark.stop()
    exit(1)
(training, test) = ratings.randomSplit([0.8, 0.2], seed=42)

# SVD via ALS (matrix factorization with implicit SVD)
als = ALS(userCol="userId", itemCol="productId", ratingCol="Rating", coldStartStrategy="drop", implicitPrefs=False)
# Tune with wider range
param_grid = ParamGridBuilder() 
param_grid = param_grid.addGrid(als.rank, [10, 20])
param_grid = param_grid.addGrid(als.regParam, [0.01, 0.1])
param_grid = param_grid.addGrid(als.maxIter, [10, 20])
param_grid = param_grid.build()
crossval = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=RegressionEvaluator(metricName="rmse", labelCol="Rating"), numFolds=3)
cv_model = crossval.fit(training)

# Predictions
predictions = cv_model.bestModel.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

# Save model
cv_model.bestModel.write().overwrite().save("svd_model")

# Precision@K (K=10): Custom eval
def precision_at_k(predictions_pd, k=10):
    def user_top_k_mean(pdf):
        top_k = pdf.nlargest(k, "prediction")
        return top_k["Rating"].mean() / 5 if len(top_k) > 0 else 0
    avg_ratings = predictions_pd.groupby("userId").apply(user_top_k_mean)
    avg_precision = avg_ratings.mean()
    baseline = 1 / predictions_pd["productId"].nunique()  # Random baseline CTR
    improvement = ((avg_precision - baseline) / baseline) * 100 if baseline > 0 else 0
    print(f"Precision@10 (proxy CTR): {avg_precision:.3f}, Improvement over random: {improvement:.1f}%")
    return avg_precision

def recall_at_k(predictions_pd, k=10, threshold=4):
    def user_recall(pdf):
        relevant = pdf[pdf['Rating'] >= threshold]
        recommended = pdf.nlargest(k, 'prediction')
        hits = recommended[recommended['Rating'] >= threshold]
        return len(hits) / len(relevant) if len(relevant) > 0 else 0
    recalls = predictions_pd.groupby('userId').apply(user_recall)
    avg_recall = recalls.mean()
    print(f"Recall@{k}: {avg_recall:.3f}")
    return avg_recall

def map_at_k(predictions_pd, k=10, threshold=4):
    def user_map(pdf):
        recommended = pdf.nlargest(k, 'prediction')
        relevant = recommended['Rating'] >= threshold
        precisions = [relevant[:i+1].mean() for i in range(len(relevant))]
        return sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    maps = predictions_pd.groupby('userId').apply(user_map)
    avg_map = maps.mean()
    print(f"MAP@{k}: {avg_map:.3f}")
    return avg_map
    def user_map(pdf):
        recommended = pdf.nlargest(k, 'prediction')
        relevant = recommended['Rating'] >= threshold
        precisions = [relevant[:i+1].mean() for i in range(len(relevant))]
        return sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    maps = predictions_pd.groupby('userId').apply(user_map)
    avg_map = maps.mean()
    print(f"MAP@{k}: {avg_map:.3f}")
    return avg_map
    # Simulate top-k recs per user, assume 'click' if rating >=4
    def user_top_k_mean(pdf):
        top_k = pdf.nlargest(k, "prediction")
        return top_k["Rating"].mean() / 5 if len(top_k) > 0 else 0

    avg_ratings = predictions_pd.groupby("userId").apply(user_top_k_mean)
    avg_precision = avg_ratings.mean()
    baseline = 1 / predictions_pd["productId"].nunique()  # Random baseline CTR
    improvement = ((avg_precision - baseline) / baseline) * 100 if baseline > 0 else 0
    print(f"Precision@10 (proxy CTR): {avg_precision:.3f}, Improvement over random: {improvement:.1f}%")
    return avg_precision

predictions_pd = predictions.toPandas()
precision_at_k(predictions_pd)
recall_at_k(predictions_pd)
map_at_k(predictions_pd)

spark.stop()