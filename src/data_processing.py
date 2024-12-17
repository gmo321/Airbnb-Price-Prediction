import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql.functions import col, sum, split, regexp_replace, avg, when, regexp_extract, size, lit, log
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler, Imputer, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import radians, col, sin, cos, atan2, sqrt




def main(input_1, input_2):
    listing_df = spark.read.parquet(input_1)

    calendar_df = spark.read.parquet(input_2)

    listing_df = listing_df.drop('listing_url', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_name', 'minimum_minimum_nights', 
                                 'maximum_maximum_nights', 'has_availability', 'availability_365', 'bathrooms', 'review_scores_accuracy', 'review_scores_cleanliness', 
                                 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value')

    calendar_df = calendar_df.drop('maximum_nights')

    listing_df = listing_df.withColumnRenamed('price', 'listing_price') \
                        .withColumnRenamed('neighbourhood_cleansed', 'neighbourhood') \
                        .withColumnRenamed('bedrooms', 'num_bedrooms') \
                        .withColumnRenamed('beds', 'num_beds') \
                        .withColumn("num_bathrooms", regexp_extract(listing_df["bathrooms_text"], r"(\d+\.?\d*)", 1).cast("float")) \
                        .withColumn("is_private_bath", when(listing_df["bathrooms_text"].like('%private%'), 1).otherwise(0)) \
                        .withColumn("is_shared_bath", when(listing_df["bathrooms_text"].like('%shared%'), 1).otherwise(0)) \
                            
    calendar_df = calendar_df.withColumnRenamed('price', 'calendar_price') 

    # Log transformation for target variable "calendar_price"
    calendar_df = calendar_df.withColumn('log_price', log('calendar_price'))

    # Using Haversine Formula to calculate proximity to city center
    # Earth's radius in kilometers
    R = 6371

    # Vancouver city center coordinates 
    vancouver_lat = 49.2827
    vancouver_lon = -123.1207

    # Add "distance_to_center_km" column using Haversine formula
    listing_df = listing_df.withColumn("lat_rad", radians(col("latitude"))) \
            .withColumn("lon_rad", radians(col("longitude"))) \
            .withColumn("vancouver_lat_rad", radians(lit(vancouver_lat))) \
            .withColumn("vancouver_lon_rad", radians(lit(vancouver_lon)))

    listing_df = listing_df.withColumn("dlat", col("vancouver_lat_rad") - col("lat_rad")) \
                        .withColumn("dlon", col("vancouver_lon_rad") - col("lon_rad"))

    listing_df = listing_df.withColumn("a", sin(col("dlat") / 2)**2 + cos(col("lat_rad")) * cos(col("vancouver_lat_rad")) * sin(col("dlon") / 2)**2) \
                    .withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1 - col("a")))) \
                    .withColumn("distance_to_center_km", R * col("c"))

    listing_df = listing_df.drop("lat_rad", "lon_rad", "vancouver_lat_rad", "vancouver_lon_rad", "dlat", "dlon", "a", "c", "latitude", "longitude")


    # Remove empty amenity columns with NULL
    listing_df = listing_df.withColumn("amenities", when(col("amenities") == "[]", lit(None)).otherwise(col("amenities")))

    # Clean and convert "amenities" string column to array
    # Add "amenities_count" as column
    listing_df = listing_df.withColumn("amenities_list", split(regexp_replace(col("amenities"), r"[\[\]\"]", ""),  ", "))

    listing_df = listing_df.drop("amenities")

    listing_df = listing_df.withColumn("amenities_count", size("amenities_list"))

    listing_df = listing_df.drop("amenities_list")


    # Joining listing_df and calendar_df
    joined_df = listing_df.join(calendar_df, listing_df.id == calendar_df.listing_id, "inner")


    joined_df = joined_df.drop('id', 'amenities', 'available', 'amenities_list', 'listing_price', 'bathrooms_text', 'date') 

    # Get average of "log_price" to be used as target variable
    aggregated_price_df = joined_df.groupBy("listing_id").agg(avg("log_price").alias("average_log_price"))

    joined_df = joined_df.join(aggregated_price_df, on="listing_id", how="inner")

    joined_df = joined_df.dropDuplicates(["listing_id"])
    

    joined_df = joined_df.drop('listing_id', 'calendar_price', 'date', 'id', 'amenity_freq_sum', 'amenity_freq_avg')


    joined_df = joined_df.filter(joined_df["average_log_price"].isNotNull())

    joined_df = joined_df.dropDuplicates()


    joined_df = joined_df.repartition(200)  

 
if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    spark = SparkSession.builder.appName('Price Prediction Model').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(input_1, input_2)