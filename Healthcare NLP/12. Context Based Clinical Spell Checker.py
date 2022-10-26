# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC <H1>Context Based Clinical Spell Checker</H1>

# COMMAND ----------

import os
import json
import string
import numpy as np
import pandas as pd

import sparknlp
import sparknlp_jsl
from sparknlp.util import *
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.pretrained import ResourceDownloader

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel


pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)

print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = RecursiveTokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")\
    .setPrefixes(["\"", "(", "[", "\n"])\
    .setSuffixes([".", ",", "?", ")","!", "'s"])

spellModel = ContextSpellCheckerModel.pretrained('spellcheck_clinical', 'en', 'clinical/models')\
    .setInputCols("token")\
    .setOutputCol("checked")

# COMMAND ----------

pipeline = Pipeline(
    stages = [
    documentAssembler,
    tokenizer,
    spellModel
  ])

empty_ds = spark.createDataFrame([[""]]).toDF("text")

lp = LightPipeline(pipeline.fit(empty_ds))

# COMMAND ----------

# MAGIC %md
# MAGIC Ok!, at this point we have our spell checking pipeline as expected. Let's see what we can do with it, see these errors,
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC _She was **treathed** with a five day course of **amoxicilin** for a **resperatory** **truct** infection._
# MAGIC 
# MAGIC _With pain well controlled on **orall** **meditation**, she was discharged to **reihabilitation** **facilitay**._
# MAGIC 
# MAGIC 
# MAGIC _Her **adominal** examination is soft, nontender, and **nonintended**_
# MAGIC 
# MAGIC _The patient was seen by the **entocrinology** service and she was discharged on 40 units of **unsilin** glargine at night_
# MAGIC       
# MAGIC _No __cute__ distress_
# MAGIC 
# MAGIC 
# MAGIC Check that some of the errors are valid English words, only by considering the context the right choice can be made.

# COMMAND ----------

example = ["She was treathed with a five day course of amoxicilin for a resperatory truct infection . ",
           "With pain well controlled on orall meditation, she was discharged to reihabilitation facilitay.",
           "Her adominal examination is soft, nontender, and nonintended.",
           "The patient was seen by the entocrinology service and she was discharged on 40 units of unsilin glargine at night",
           "No cute distress",
          ]

for pairs in lp.annotate(example):

  print (list(zip(pairs['token'],pairs['checked'])))

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #