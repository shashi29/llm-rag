# import os
# import pyarrow.parquet as pq
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, NGram
# from pyspark.ml.clustering import KMeans
# from pyspark.ml import Pipeline
# import matplotlib.pyplot as plt
# from pyspark.sql import functions as F
# from tqdm import tqdm  # Import tqdm for progress tracking
# import time

# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Parquet Reader") \
#     .config("spark.driver.memory", "12g") \
#     .config("spark.executor.memory", "12g") \
#     .getOrCreate()

# # Step 0: Start tracking time
# start_time = time.time()

# # Read Parquet data
# df = spark.read.parquet("/root/data/ProcessedResults/parquet_consolidated/*")

# # Track time for reading the parquet
# parquet_read_time = time.time()
# print(f"Time taken to read Parquet data: {parquet_read_time - start_time:.2f} seconds")

# # Step 1: Text Preprocessing with Bigrams and Trigrams

# # Tokenize the text
# tokenizer = Tokenizer(inputCol="content", outputCol="words")

# # Remove stop words
# remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# # Generate bigrams
# bigram = NGram(n=2, inputCol="filtered_words", outputCol="bigrams")

# # Generate trigrams
# trigram = NGram(n=3, inputCol="filtered_words", outputCol="trigrams")

# # Concatenate bigrams and trigrams
# concat_col = F.concat_ws(" ", F.col("filtered_words"), F.col("bigrams"), F.col("trigrams"))

# # Create TF-IDF features from bigrams and trigrams
# hashingTF = HashingTF(inputCol="trigrams", outputCol="rawFeatures", numFeatures=5000)
# idf = IDF(inputCol="rawFeatures", outputCol="features")

# # Create a pipeline with the text preprocessing steps
# pipeline = Pipeline(stages=[tokenizer, remover, bigram, trigram, hashingTF, idf])

# # Fit the pipeline to the data
# pipeline_start_time = time.time()
# model = pipeline.fit(df)
# preprocessed_df = model.transform(df)

# # Track pipeline processing time
# pipeline_end_time = time.time()
# print(f"Time taken for text preprocessing: {pipeline_end_time - pipeline_start_time:.2f} seconds")

# # Step 2: Elbow Method to Find Optimal k
# wcss = []  # To store WCSS for each k
# k_values = range(5, 11)  # Test k from 5 to 10

# # Initialize time tracking for KMeans
# kmeans_start_time = time.time()

# # Use tqdm to track progress in the loop
# for k in tqdm(k_values, desc="KMeans Clustering", unit="cluster"):
#     kmeans_iteration_start_time = time.time()
#     kmeans = KMeans(k=k, seed=1, featuresCol="features", maxIter=50)  # Set max iterations
#     kmeans_model = kmeans.fit(preprocessed_df)
#     wcss.append(kmeans_model.summary.trainingCost)
    
#     # Optional: If you want to print time for each iteration
#     elapsed = time.time() - kmeans_iteration_start_time
#     print(f"KMeans for k={k} completed in {elapsed:.2f} seconds")

# # Track KMeans time
# kmeans_end_time = time.time()
# print(f"Total time for KMeans clustering: {kmeans_end_time - kmeans_start_time:.2f} seconds")

# # Step 3: Plot the Elbow Method Results
# plot_start_time = time.time()

# plt.figure(figsize=(10, 6))
# plt.plot(k_values, wcss, marker='o')
# plt.title("Elbow Method for Optimal K")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("WCSS")
# plt.xticks(k_values)
# plt.grid()

# # Save the plot
# plt.savefig("elbow_method_optimal_k.png", bbox_inches='tight')  # Save the figure
# plt.close()  # Close the plot to free up memory

# # Track plotting time
# plot_end_time = time.time()
# print(f"Time taken to plot and save the elbow method: {plot_end_time - plot_start_time:.2f} seconds")

# # Step 4: Overall time tracking
# total_end_time = time.time()
# print(f"Total time for the entire process: {total_end_time - start_time:.2f} seconds")

import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, NGram
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import time
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Parquet Reader") \
    .config("spark.driver.memory", "14g") \
    .config("spark.executor.memory", "14g") \
    .getOrCreate()

# Start tracking time
start_time = time.time()

# Read Parquet data
df = spark.read.parquet("/root/data/ProcessedResults/parquet_consolidated/*")

# Track time for reading the parquet
parquet_read_time = time.time()
print(f"Time taken to read Parquet data: {parquet_read_time - start_time:.2f} seconds")

# Text Preprocessing with Bigrams and Trigrams
tokenizer = Tokenizer(inputCol="content", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
bigram = NGram(n=2, inputCol="filtered_words", outputCol="bigrams")
trigram = NGram(n=3, inputCol="filtered_words", outputCol="trigrams")
concat_col = F.concat_ws(" ", F.col("filtered_words"), F.col("bigrams"), F.col("trigrams"))
hashingTF = HashingTF(inputCol="trigrams", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Create a pipeline with the text preprocessing steps
pipeline = Pipeline(stages=[tokenizer, remover, bigram, trigram, hashingTF, idf])

# Fit the pipeline to the data
pipeline_start_time = time.time()
model = pipeline.fit(df)
preprocessed_df = model.transform(df)

# Track pipeline processing time
pipeline_end_time = time.time()
print(f"Time taken for text preprocessing: {pipeline_end_time - pipeline_start_time:.2f} seconds")

# KMeans Clustering with k=9
k = 9
kmeans_start_time = time.time()
kmeans = KMeans(k=k, seed=1, featuresCol="features", maxIter=50)
kmeans_model = kmeans.fit(preprocessed_df)

# Track KMeans time
kmeans_end_time = time.time()
print(f"Time taken for KMeans clustering (k={k}): {kmeans_end_time - kmeans_start_time:.2f} seconds")

# Apply the model to get cluster assignments
clustered_df = kmeans_model.transform(preprocessed_df)

# Select the required columns: key, filtered_words, bigrams, trigrams, prediction
selected_columns_df = clustered_df.select("key", "filtered_words", "bigrams", "trigrams", "prediction")

# Convert the selected PySpark DataFrame to Pandas DataFrame
selected_columns_pandas = selected_columns_df.toPandas()

# Write the Pandas DataFrame to an Excel file
excel_path = "cluster_results.xlsx"
selected_columns_pandas.to_excel(excel_path, index=False)
print(f"Data written to {excel_path}")

# Overall time tracking
total_end_time = time.time()
print(f"Total time for the entire process: {total_end_time - start_time:.2f} seconds")

# Stop Spark session
spark.stop()
