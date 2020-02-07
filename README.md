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
First, I read parkinsons.names to learn about the data set and ran printSchema to verify that the data matches description. The updrs dataset
is from different individuals who all have Parkinson's, so it is of no value in this endeavor. The major issue with this
data is the low number of distinct individuals (32). The high number of variables to consider helps, but also adds strongly to the possibility of
overfitting. Most importantly, the same individual cannot be in both the training and test sets, or the results will not
generalize to people who were not part of the original study.

The training and test sets were thus created by taking 80% of the patients (rounded down) at random and using all samples from
these patients as the training set. The samples from the remaining patients were used as the test set.

To avoid overfitting, the maximum depth of each tree was limited to 5. Experimentation showed that the results stopped improving
after the number of trees exceeded 10.






## Citations
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 
'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', 
IEEE Transactions on Biomedical Engineering
