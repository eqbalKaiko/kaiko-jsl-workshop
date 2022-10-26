# Databricks notebook source
# MAGIC %md
# MAGIC # Create a Database of Oncological Entities Based on Unstructured Notes
# MAGIC In this notebook, we create a database of entities based on extracted terms from the notes in previous notebooks. 
# MAGIC This database can be used for dashboarding using [Databricks SQL](https://databricks.com/product/databricks-sql).

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Initial configurations

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

delta_path='/FileStore/HLS/nlp/delta/jsl/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Temporary Views

# COMMAND ----------

spark.read.load(f"{delta_path}/silver/icd10-hcc-df").createOrReplaceTempView('icd10Hcc')
spark.read.load(f"{delta_path}/gold/best-icd-mapped").createOrReplaceTempView('bestIcdMapped')
spark.read.load(f"{delta_path}/gold/rxnorm-res-cleaned").createOrReplaceTempView('rxnormRes')
spark.read.load(f"{delta_path}/silver/rxnorm-code-greedy-res").createOrReplaceTempView('rxnormCodeGreedy')
spark.read.load(f"{delta_path}/silver/temporal-re").createOrReplaceTempView('temporalRe')
spark.read.load(f"{delta_path}/silver/bodypart-relationships").createOrReplaceTempView('bodypartRelationships')
spark.read.load(f"{delta_path}/silver/cpt").createOrReplaceTempView('cpt')
spark.read.load(f"{delta_path}/silver/assertion").createOrReplaceTempView('assertion')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the Database

# COMMAND ----------

database_name='jsl_onc'
DatabaseName=''.join([st.capitalize() for st in database_name.split('_')])
database_path=f"{delta_path}tables/{database_name}"
print(f"{DatabaseName} database tables will be stored in {database_path}")

# COMMAND ----------

sql(f"DROP DATABASE IF EXISTS {DatabaseName} CASCADE;")
sql(f"CREATE DATABASE IF NOT EXISTS {DatabaseName} LOCATION '{database_path}'")
sql(f"USE {DatabaseName};")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE Rxnorm_Res AS (
# MAGIC   select md5(path) as note_id,path,confidence, drug_chunk,rxnorm_code,drugs as drug from rxnormRes
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE CPT AS (
# MAGIC   select md5(path) as note_id, path, confidence, chunks, entity,cpt_code,cpt
# MAGIC from cpt)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ASSERTION AS (
# MAGIC   select md5(path) as note_id, path, chunk, entity,assertion from assertion
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE TEMPORAL_RE AS (
# MAGIC   select md5(path) as note_id, * from temporalRe
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE BEST_ICD AS (
# MAGIC   select * from bestIcdMapped
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ICD10_HCC AS (
# MAGIC   select md5(path) as note_id, path, confidence, final_chunk, entity,icd10_code,icd_codes_names,icd_code_billable
# MAGIC   from icd10Hcc
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ICD10_HCC

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.