# CMPT 732 Final Project: Vancouver Airbnb Data Analysis

To run the project locally:
Clone the repository:
git clone github.sfu.ca/sxa34/404

# Install required dependencies
pip install -r requirements.txt

## ETL
ETL is performed on raw data which is stored in raw_data directory. The .lz4 parquet files are stored in raw_data directory: calendar_parquet, listings_parquet, reviews_parquet. The ETL was processed through HDFS file storage and was executed on a cluster using Apache Spark.

## Price Prediction Model using RandomForestRegressor
Price prediction model that forecasts the optimal price of listings for the coming year, using historical data and features that influence pricing, such as amenities, location, seasonality, and listing characteristics.

### Code running
    spark-submit price_pred_model.py listings_parquet calendar_parquet