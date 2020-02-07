import pyspark
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, IndexToString
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('data/parkinsons.data',
                        header=True, inferSchema=True)

labelIndexer = StringIndexer(inputCol="name", outputCol="label").fit(df)
cols=['MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)']
assembler_features = VectorAssembler(inputCols=cols, outputCol='features')

print assembler_features

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="status", featuresCol="features", numTrees=10)


# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler_features, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)


predictions = model.transform(df)

predictions.show()
