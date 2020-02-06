import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('data/parkinsons.data',
                        header=True, inferSchema=True)
df.printSchema()
