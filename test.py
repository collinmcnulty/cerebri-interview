import pyspark
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, IndexToString
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd
data_path = __file__ + '/../data/parkinsons.data'
pand = pd.read_csv(data_path)
a = pand['name'].str.split('_', expand=True)[2].unique()



spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv(data_path,
                        header=True, inferSchema=True)

split_col = pyspark.sql.functions.split(df['name'], '_')
df = df.withColumn('patient', split_col.getItem(2))
labelIndexer = StringIndexer(inputCol="status", outputCol="label").fit(df)
cols=['MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)']
assembler_features = VectorAssembler(inputCols=cols, outputCol='features')


# Split the data into training and test sets (20% held out for testing)
(trainingData, testData) = df.randomSplit([0.8, 0.2])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="status", featuresCol="features", numTrees=10)


# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler_features, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)


predictions = model.transform(df)
predictions.printSchema()
# predictions = predictions.withColumnRenamed('status', 'label')

truncated = predictions.select(['prediction', 'label'])
truncated.show()
metrics = BinaryClassificationMetrics(truncated.rdd)
m_metrics = MulticlassMetrics(truncated.rdd)
1
#
# def evaluate_results(predictions):
#     metrics = ['Precision', 'Recall', 'AUC']
#     values = []
#     metrics_evaluations = ['precision', 'recall', 'areaUnderROC']
#     evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="status")
#     for i, metric in enumerate(metrics_evaluations):
#         val = evaluator.setMetricName(metric).evaluate(predictions)
#         print val
#         values.append(val)
#     results = pd.DataFrame({'metric': metrics,
#                   'value': values})
#     return results
#
# results = evaluate_results(predictions)
# results.to_parquet('ml_test')
# predictions.show()
