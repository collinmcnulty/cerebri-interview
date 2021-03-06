from __future__ import division

import random

import pandas as pd
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt

# Set Configurable Elements

cols = 'MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),Jitter:DDP,MDVP:Shimmer,MDVP:APQ,Shimmer:DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2'
NUM_TREES = 8
MAX_DEPTH = 5
TRAINING_RATIO = 0.8

random.seed(1)

# Load Data
data_path = __file__ + '/../data/parkinsons.data'
spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv(data_path,
                    header=True, inferSchema=True)

labelIndexer = StringIndexer(inputCol="status", outputCol="label").fit(df)

cols = cols.split(',')
features = VectorAssembler(inputCols=cols, outputCol='features')

# Split the data into training and test sets (20% held out for testing)

# Identify individual patients
split_col = pyspark.sql.functions.split(df['name'], '_')
df = df.withColumn('patient', split_col.getItem(2))

# Select training patients at random
number_of_training_patients = int(df.select('patient').distinct().count() * TRAINING_RATIO)
training_patients = df.select('patient').distinct().orderBy(rand(seed=1)).limit(number_of_training_patients)

# Divide into training and test data
trainingData = df.join(training_patients, ['patient'], 'inner')
testData = df.join(training_patients, ['patient'], 'leftanti')

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=NUM_TREES, maxDepth=MAX_DEPTH)

# Make pipeline from the stages
pipeline = Pipeline(stages=[labelIndexer, features, rf])

# Train model
model = pipeline.fit(trainingData)

# Make prediction for the test set
predictions = model.transform(testData)


def get_metrics(predictions):
    auc = BinaryClassificationEvaluator().evaluate(predictions)

    truncated = predictions.select(['prediction', 'probability', 'label'])
    truncated_pandas = truncated.toPandas()
    true_positive = ((truncated_pandas['prediction'] == 1) & (truncated_pandas['label'] == 1)).sum()
    false_positive = ((truncated_pandas['prediction'] == 1) & (truncated_pandas['label'] == 0)).sum()
    true_negative = ((truncated_pandas['prediction'] == 0) & (truncated_pandas['label'] == 0)).sum()
    false_negative = ((truncated_pandas['prediction'] == 0) & (truncated_pandas['label'] == 1)).sum()

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return truncated, precision, recall, auc


truncated, precision, recall, auc = get_metrics(predictions)
truncated.show(50)

# Store Results
results = pd.DataFrame({'metric': ['Precision', 'Recall', 'AUC'],
                        'value': [precision, recall, auc]})

results = spark.createDataFrame(results)
results = results.withColumn('value', results['value'].cast('float'))
results.write.parquet('ml_test', compression=None, mode='overwrite')

final = spark.read.parquet('ml_test')
final.printSchema()
final.show()
1

# Below considers all samples to be independent, which is absolutely not true
(trainingData, testData) = df.randomSplit([0.8, 0.2])
false_model = pipeline.fit(trainingData)
false_predictions = false_model.transform(testData)

truncated, precision, recall, auc = get_metrics(false_predictions)
false_results = pd.DataFrame({'metric': ['Precision', 'Recall', 'AUC'],
                              'value': [precision, recall, auc]})
print false_results
1
