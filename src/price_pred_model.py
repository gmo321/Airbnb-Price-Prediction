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

    # Split the data
    train_data, test_data = joined_df.randomSplit([0.80, 0.20], seed=42)
    train_data = train_data.cache()
    test_data = test_data.cache()

    #train_data.show()

    numeric_columns = ["accommodates",
            "num_bedrooms",
            "num_beds",
            "num_bathrooms",
            "number_of_reviews",
            "review_scores_rating",
            "reviews_per_month",
            "distance_to_center_km",
            "amenities_count",
            "minimum_nights"]
    
    categorical_columns = ["neighbourhood",
                   "property_type",
                   "is_private_bath",
                   "is_shared_bath",
                   "room_type"]


    # Numerical columns
    imputer = Imputer(
        inputCols=numeric_columns,
        outputCols=[num_col+"_imputed" for num_col in numeric_columns],
        strategy="mean"
    )


    numeric_assembler = VectorAssembler(
        inputCols=[num_col+"_imputed" for num_col in numeric_columns],
        outputCol="numeric_features")

    
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="numeric_features_scaled",
            withStd=True,
            withMean=False)


    # Categorical columns 
    indexer = StringIndexer(
        inputCols=categorical_columns,
        outputCols=[cat_col+"_index" for cat_col in categorical_columns],
        handleInvalid="keep")

    
    encoder = OneHotEncoder(inputCols=[cat_col+"_index" for cat_col in categorical_columns], 
                            outputCols=[cat_col+"_ohe" for cat_col in categorical_columns])


    assembler = VectorAssembler(inputCols=["numeric_features_scaled"] +
                                           [cat_col+"_ohe" for cat_col in categorical_columns],
                                outputCol="features")
    

    rf = RandomForestRegressor(featuresCol="features", numTrees=15, maxDepth=20, labelCol="average_log_price")


    pipeline = Pipeline(stages=[imputer, numeric_assembler, scaler, indexer, encoder, assembler, rf])


    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)

    predictions.select("prediction", "average_log_price", "features").show(10)
    

    # Evaluate the predictions
    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='average_log_price', metricName='r2')
    
    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='average_log_price', metricName='rmse')

    # Evaluate on Training Data
    train_predictions = model.transform(train_data)
    train_r2 = r2_evaluator.evaluate(train_predictions)
    train_rmse = rmse_evaluator.evaluate(train_predictions)

    # Evaluate on Testing Data
    test_predictions = model.transform(test_data)
    test_r2 = r2_evaluator.evaluate(test_predictions)
    test_rmse = rmse_evaluator.evaluate(test_predictions)

    # Print Results
    print('Training R² =', train_r2)
    print('Training RMSE =', train_rmse)
    print('Testing R² =', test_r2)
    print('Testing RMSE =', test_rmse)


    # Evaluate feature importances
    rf_model = model.stages[-1]  
    importances = rf_model.featureImportances

    importances_dense = importances.toArray()

    feature_names = [num_col+"_imputed" for num_col in numeric_columns] + ["neighbourhood_ohe", 
        "room_type_ohe", "property_type_ohe", "is_private_bath_ohe", "is_shared_bath_ohe"]

    # Create a DataFrame with features and their corresponding importances
    feature_importances = pd.DataFrame(
        sorted(list(zip(feature_names, importances_dense)), key=lambda x: abs(x[1]), reverse=True),
        columns=['feature', 'importance'])

    # Bar graph of feature importances
    feature_importances.plot.barh(x='feature', y='importance', legend=False)
    plt.xlabel('Importance')
    plt.title('Feature Importances (Random Forest)')
    plt.show()

    

    # Visualization of Actual vs Predicted Prices
    predictions_pd = predictions.select("average_log_price", "prediction").toPandas()

    actual = predictions_pd["average_log_price"].values  
    predicted = predictions_pd["prediction"].values

    plot_min = min(actual.min(), predicted.min())  
    plot_max = max(actual.max(), predicted.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5, color="skyblue", label="Predicted vs. Actual")

    # Add the reference line y = x
    plt.plot([plot_min, plot_max], [plot_min, plot_max], color="red", linestyle="--", label="Reference Line (y = x)")

    plt.title("Predicted vs. Actual Prices", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Average Log Price", fontsize=12)
    plt.ylabel("Predicted Average Log Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()

 
if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    spark = SparkSession.builder.appName('Price Prediction Model').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(input_1, input_2)