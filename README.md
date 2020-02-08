# cerebri interview


## Compilation
Pyspark code does not need to be compiled, see Environment.

## Environment
Prepare a fresh environment by running `conda create -n ml_test_env pandas pyspark python=2.7`. Activate this new environment
with `conda activate ml_test-env`.

## Spark Submit
Unzip this folder anywhere on your computer. Navigate (cd) to the folder in command prompt or terminal and point spark-submit
to test.py. On Windows, the command should be `%SPARK_HOME%/bin/spark-submit test.py`.


## Design
First, I read parkinsons.names to learn about the data set and ran printSchema to verify that the data matches description. The updrs dataset
is from different individuals who all have Parkinson's, so it is of no value in this endeavor. The major issue with this
data is the low number of distinct individuals (32). The high number of variables to consider helps, but also adds strongly to the possibility of
overfitting. Most importantly, the same individual cannot be in both the training and test sets, or the results will not
generalize to people who were not part of the original study.

Dimensional reduction was performed by observing the scatter matrix of the data and removing variables which had extremely
strong correlations with another variable (leaving only one variable of the strongly correlated set). This resulted in the removal
of most of the "jitter" set and most of the "shimmer" set, plus the removal of "PPE"

The training and test sets were thus created by taking 80% of the patients (rounded down) at random and using all samples from
these patients as the training set. The samples from the remaining patients were used as the test set.

To avoid overfitting, the maximum depth of each tree was limited to 5. Experimentation showed that the results stopped improving
after the number of trees exceeded 8.


## Possible Extensions
Frankly, the results are not very powerful; the low sample size makes this extremely challenging. The clearest path to 
improvement would be to make use of the fact that each patient gave multiple sample and combine their measurements from each
sample to create new features. This would require that new patients who want to be evaluated for Parkinson's give multiple samples
before a prediction can be made, but it may produce better results than have been achieved here.



## Citations
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 
'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', 
IEEE Transactions on Biomedical Engineering
