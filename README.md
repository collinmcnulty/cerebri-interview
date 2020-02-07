# cerebri interview


# Compilation
Pyspark code does not need to be compiled, see Environment.

# Environment
Prepare a fresh environment by running `conda create -n ml_test_env -f test-env.yml`. Activate this new environment
with `conda activate test-env`.

# Spark Submit
Unzip this folder anywhere on your computer. Navigate (cd) to the folder in command prompt or terminal and point spark-submit
to test.py. On Windows, the command should be `%SPARK_HOME%/bin/spark-submit test.py`.


# Design
First, I read parkinsons.names to learn about the data set and ran printSchema to verify data matches description.



## Citations
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 
'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', 
IEEE Transactions on Biomedical Engineering
