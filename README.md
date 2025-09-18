# Product Recommender System with SVD (Spark ALS)

## Problem Statement
Build a robust recommender system for an online retail dataset to predict user ratings for products and recommend top items. The goal is to handle cold-start issues, tune model performance, and provide meaningful evaluation metrics.

## Models Used
- **Alternating Least Squares (ALS)**: Matrix factorization algorithm implemented in Spark MLlib. ALS is chosen for its scalability, ability to handle implicit/explicit feedback, and suitability for large, sparse datasets.
- **Evaluation Metrics**: RMSE (Root Mean Squared Error), Precision@K, Recall@K, MAP@K for ranking quality.

## Why ALS?
- **Scalability**: Handles large datasets efficiently using distributed computing.
- **Cold-Start Handling**: Can filter out users/products with few ratings.
- **Flexibility**: Supports both implicit and explicit feedback.
- **Industry Standard**: Widely used in production recommender systems.

## Model Architecture
- **Input**: User-Product-Rating data (from `online_retail.csv`).
- **Preprocessing**: Aggregate ratings, filter cold-start users/products, output to Parquet.
- **ALS Training**: Hyperparameter tuning (rank, regParam, maxIter) via cross-validation.
- **Evaluation**: RMSE, Precision@K, Recall@K, MAP@K.
- **Output**: Trained ALS model, recommendations, metrics.

## How the Model Works
1. **Data Preparation** (`data_prep.py`):
	- Loads `online_retail.csv`.
	- Aggregates ratings per user/product.
	- Filters out users/products with fewer than 5 ratings.
	- Saves processed ratings to `ratings.parquet`.
2. **Model Training & Evaluation** (`model_train.py`):
	- Loads `ratings.parquet`.
	- Splits data into training/test sets.
	- Tunes ALS hyperparameters using cross-validation.
	- Trains ALS model and evaluates with RMSE, Precision@K, Recall@K, MAP@K.
	- Saves best model.
3. **Web App** (`app.py`):
	- Streamlit app for interactive recommendations and visualization.

## Code Workflow (Step-by-Step)
1. **Prepare Data**
	- Run: `python data_prep.py`
	- Output: `ratings.parquet`
2. **Train Model & Evaluate**
	- Run: `python model_train.py`
	- Output: Model metrics, saved model
3. **Launch Web App**
	- Run: `streamlit run app.py`
	- Output: Interactive UI in browser

## How to Run in GitHub Codespaces
1. **Clone the Repository**
	```bash
	git clone https://github.com/ksp7273/product-recommender-svd-spark.git
	cd product-recommender-svd-spark
	```
2. **Install Dependencies**
	- Ensure Python 3.12 and pip are available.
	- Install required packages:
	  ```bash
	  pip install pyspark pandas scikit-learn streamlit
	  ```
3. **Prepare Data**
	```bash
	python data_prep.py
	```
4. **Train Model**
	```bash
	python model_train.py
	```
5. **Run Web App**
	```bash
	streamlit run app.py
	```
	- Open the provided URL in your browser to view the app.

## File Overview
- `app.py`: Streamlit web app for recommendations.
- `data_prep.py`: Data preprocessing and cold-start filtering.
- `model_train.py`: ALS model training, hyperparameter tuning, evaluation.
- `online_retail.csv`: Input data file (user-product-rating).

## Notes
- For best results, ensure `online_retail.csv` is diverse and realistic.
- The pipeline is robust to cold-start and missing data issues.
- All code is compatible with GitHub Codespaces and standard Python environments.