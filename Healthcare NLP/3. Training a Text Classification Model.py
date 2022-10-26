# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Trainining a Text Classification Model

# COMMAND ----------

import os
import json
import string
import numpy as np
import pandas as pd


import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.util import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.pretrained import ResourceDownloader

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)


print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset

# COMMAND ----------

 !wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/petfinder-mini.csv
 
 dbutils.fs.cp("file:/databricks/driver/petfinder-mini.csv", "dbfs:/") 

# COMMAND ----------

dataframe = pd.read_csv('petfinder-mini.csv')

# COMMAND ----------

# In the original dataset "4" indicates the pet was not adopted.

dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)

# COMMAND ----------

dataframe = dataframe.drop(['AdoptionSpeed'], axis=1)

# COMMAND ----------

dataframe.head()

# COMMAND ----------

dataframe.columns

# COMMAND ----------

dataframe.info()

# COMMAND ----------

dataframe.target.value_counts()

# COMMAND ----------

dataframe.Description = dataframe.Description.fillna('- no description -')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurize with Sklearn Column Transformer

# COMMAND ----------

from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

column_trans = make_column_transformer(
     (OneHotEncoder(handle_unknown='ignore', categories='auto'), ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
       'FurLength', 'Vaccinated', 'Sterilized', 'Health']),
     (TfidfVectorizer(max_features=100,  norm='l2', ngram_range=(1, 3)), 'Description'),
     remainder=StandardScaler())

X = column_trans.fit_transform(dataframe.drop(['target'], axis=1))

y = dataframe.target

# COMMAND ----------

y.nunique()


# COMMAND ----------

X.shape

# COMMAND ----------

input_dim = X.shape[1]

# COMMAND ----------

input_dim

# COMMAND ----------

import scipy.sparse

df = pd.DataFrame.sparse.from_spmatrix(X)

df.columns = ['col_{}'.format(i) for i in range(input_dim)]

df = df.fillna(0)

df['target']= y

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with Spark NLP Generic Classifier

# COMMAND ----------

# MAGIC %md
# MAGIC **Building a pipeline**
# MAGIC 
# MAGIC The FeaturesAssembler is used to collect features from different columns. It can collect features from single value columns (anything which can be cast to a float, if casts fails then the value is set to 0), array columns or SparkNLP annotations (if the annotation is an embedding, it takes the embedding, otherwise tries to cast the 'result' field). The output of the transformer is a FEATURE_VECTOR annotation (the numeric vector is in the 'embeddings' field).
# MAGIC 
# MAGIC The GenericClassifierApproach takes FEATURE_VECTOR annotations as input, classifies them and outputs CATEGORY annotations. The operation of the classifier is controled by the following methods:
# MAGIC 
# MAGIC *setEpochsNumber(int)* - Determines how many epochs the model is trained.
# MAGIC 
# MAGIC *setBatchSize(int)* - Sets the batch size during training.
# MAGIC 
# MAGIC *setLearningRate(float)* - Sets the learning rate.
# MAGIC 
# MAGIC *setValidationSplit(float)* - Sets the proportion of examples in the training set used for validation.
# MAGIC 
# MAGIC *setModelFile(string)* - Loads a model from the specified location and uses it instead of the default model.
# MAGIC 
# MAGIC *setFixImbalance(boolean)* - If set to true, it tries to balance the training set by weighting the classes according to the inverse of the examples they have.
# MAGIC 
# MAGIC *setFeatureScaling(string)* - Normalizes the feature factors using the specified method ("zscore", "minmax" or empty for no normalization).
# MAGIC 
# MAGIC *setOutputLogsPath(string)* - Sets the path to a folder where logs of training progress will be saved. No logs are generated if no path is specified.

# COMMAND ----------

spark_df = spark.createDataFrame(df)
spark_df.select(spark_df.columns[-10:]).show(2)

# COMMAND ----------

(training_data, test_data) = spark_df.randomSplit([0.8, 0.2], seed = 100)

print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC Training with a custom graph

# COMMAND ----------

#or just use the one we already have in the repo
'''
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/generic_classifier_graph/pet.in1202D.out2.pb -P /databricks/driver/gc_graph
'''

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/gc_graph

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/generic_logs

# COMMAND ----------

# MAGIC %md
# MAGIC Graph and log folder has been created. <br/>
# MAGIC We will create a custom graph and train the model. <br/>
# MAGIC **WARNING:** For training a Generic Classifier with custom graph, please use TensorFlow version 2.3

# COMMAND ----------

import tensorflow
from sparknlp_jsl.training import tf_graph

# COMMAND ----------

DL_params = {"input_dim": input_dim, 
             "output_dim": y.nunique(), 
             "hidden_layers": [300, 200, 100], 
             "hidden_act": "tanh",
             'hidden_act_l2':1,
             'batch_norm':1}


tf_graph.build("generic_classifier",
               build_params=DL_params, 
               model_location="file:/dbfs/gc_graph",
               model_filename="auto")

# COMMAND ----------

from sparknlp_jsl.base import *


features_asm = FeaturesAssembler()\
      .setInputCols(['col_{}'.format(i) for i in range(X.shape[1])])\
      .setOutputCol("features")

gen_clf = GenericClassifierApproach()\
    .setLabelColumn("target")\
    .setInputCols(["features"])\
    .setOutputCol("prediction")\
    .setModelFile('file:/dbfs/gc_graph/gcl.302.2.pb')\
    .setEpochsNumber(50)\
    .setBatchSize(100)\
    .setFeatureScaling("zscore")\
    .setFixImbalance(True)\
    .setLearningRate(0.001)\
    .setOutputLogsPath("dbfs:/generic_logs")\
    .setValidationSplit(0.2) # keep 20% of the data for validation purposes

clf_Pipeline = Pipeline(stages=[
    features_asm, 
    gen_clf])

# COMMAND ----------

clf_model = clf_Pipeline.fit(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Getting predictions

# COMMAND ----------

pred_df = clf_model.transform(test_data)

# COMMAND ----------

pred_df.select('target','prediction.result').show()

# COMMAND ----------

preds_df = pred_df.select('target','prediction.result').toPandas()

# Let's explode the array and get the item(s) inside of result column out
preds_df['result'] = preds_df['result'].apply(lambda x : int(x[0]))

# COMMAND ----------

# We are going to use sklearn to evalute the results on test dataset
from sklearn.metrics import classification_report, accuracy_score

print (classification_report(preds_df['result'], preds_df['target'], digits=4))

print (accuracy_score(preds_df['result'], preds_df['target']))

# COMMAND ----------

# MAGIC %md
# MAGIC # Case Study: Alexa Review Classification

# COMMAND ----------

! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/amazon_alexa.tsv -P /dbfs/

# COMMAND ----------

import pandas as pd
df = pd.read_csv('/dbfs/amazon_alexa.tsv', sep='\t')
df

# COMMAND ----------

df.verified_reviews = df.verified_reviews.str.lower()

# COMMAND ----------

df.feedback.value_counts()

# COMMAND ----------

from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

column_trans = make_column_transformer(
     (TfidfVectorizer(max_features=1000,  norm='l2', ngram_range=(1, 3)), 'verified_reviews'))

X = column_trans.fit_transform(df.drop(['feedback'], axis=1))

y = df.feedback

# COMMAND ----------

import scipy.sparse

sdf = pd.DataFrame.sparse.from_spmatrix(X)

sdf.columns = ['col_{}'.format(i) for i in range(X.shape[1])]

sdf = sdf.fillna(0)

sdf['feedback']= y

sdf.head()

# COMMAND ----------

input_spark_df = spark.createDataFrame(sdf)

# COMMAND ----------

(training_data, test_data) = input_spark_df.randomSplit([0.8, 0.2], seed = 100)

print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))

# COMMAND ----------

from sparknlp_jsl.base import *

features_asm = FeaturesAssembler()\
      .setInputCols(['col_{}'.format(i) for i in range(X.shape[1])])\
      .setOutputCol("features")
        
gen_clf = GenericClassifierApproach()\
    .setLabelColumn("feedback")\
    .setInputCols(["features"])\
    .setOutputCol("prediction")\
    .setEpochsNumber(50)\
    .setBatchSize(100)\
    .setFeatureScaling("zscore")\
    .setFixImbalance(True)\
    .setLearningRate(0.001)\
    .setOutputLogsPath("file:/databricks/driver/generic_logs")\
   #.setModelFile("/databricks/driver/gc_graph/pet_in1202D_out2.pb")
    

clf_Pipeline = Pipeline(stages=[
    features_asm, 
    gen_clf])

clf_model = clf_Pipeline.fit(training_data)


# COMMAND ----------

pred_df = clf_model.transform(test_data)

preds_df = pred_df.select('feedback','prediction.result').toPandas()

# Let's explode the array and get the item(s) inside of result column out
preds_df['result'] = preds_df['result'].apply(lambda x : int(x[0]))

# We are going to use sklearn to evalute the results on test dataset
from sklearn.metrics import classification_report, accuracy_score

print (classification_report(preds_df['result'], preds_df['feedback'], digits=4))

print (accuracy_score(preds_df['result'], preds_df['feedback']))
