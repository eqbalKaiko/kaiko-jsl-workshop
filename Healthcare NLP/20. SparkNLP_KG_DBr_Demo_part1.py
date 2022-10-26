# Databricks notebook source
# MAGIC %md
# MAGIC # Patient Cohort Building with NLP and Knowledge Graphs

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will build a Knowledge Graph (KG) using Spark NLP relation extraction models and Neo4j. The main point of this notebook is to show creating a clinical knowledge graph using Spark NLP pretrained models. For this purpose, we will use pretrained relation extraction and NER models. After creating the knowledge graph, we will query the KG to get some insightful results.

# COMMAND ----------

# MAGIC %md
# MAGIC [Cluster Setup](https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-on-databricks)

# COMMAND ----------

# MAGIC %md
# MAGIC **Initial Configurations**

# COMMAND ----------

import json
import os

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel,Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import *

from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl
import sparknlp

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth",100)

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

spark._jvm.com.johnsnowlabs.util.start.registerListenerAndStartRefresh()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Medical Dataset
# MAGIC 
# MAGIC In this notebook, we will use the medical records in csv format.

# COMMAND ----------

notes_path='/FileStore/HLS/kg/data/'
delta_path='/FileStore/HLS/kg/delta/jsl/'

dbutils.fs.mkdirs(notes_path)
os.environ['notes_path']=f'/dbfs{notes_path}'

# COMMAND ----------

# MAGIC %sh
# MAGIC cd $notes_path
# MAGIC !wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/databricks/python/healthcare_case_studies/data/data.csv

# COMMAND ----------

dbutils.fs.ls(f'{notes_path}/')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data and Write to Bronze Delta Layer
# MAGIC 
# MAGIC There are 965 clinical records stored in delta table. We read the data and write the records into bronze delta tables.

# COMMAND ----------

df = pd.read_csv(f'{notes_path}/data.csv', sep=';')
df

# COMMAND ----------

sparkDF=spark.createDataFrame(df) 
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

sparkDF.write.format('delta').mode('overwrite').save(f'{delta_path}/bronze/dataset')
display(dbutils.fs.ls(f'{delta_path}/bronze/dataset'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Posology RE Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Posology Releation Extraction
# MAGIC 
# MAGIC Posology relation extraction pretrained model supports the following relatios:
# MAGIC 
# MAGIC DRUG-DOSAGE
# MAGIC DRUG-FREQUENCY
# MAGIC DRUG-ADE (Adversed Drug Events)
# MAGIC DRUG-FORM
# MAGIC DRUG-ROUTE
# MAGIC DRUG-DURATION
# MAGIC DRUG-REASON
# MAGIC DRUG=STRENGTH
# MAGIC 
# MAGIC The model has been validated against the posology dataset described in (Magge, Scotch, & Gonzalez-Hernandez, 2018).
# MAGIC 
# MAGIC | Relation | Recall | Precision | F1 | F1 (Magge, Scotch, & Gonzalez-Hernandez, 2018) |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | DRUG-ADE | 0.66 | 1.00 | **0.80** | 0.76 |
# MAGIC | DRUG-DOSAGE | 0.89 | 1.00 | **0.94** | 0.91 |
# MAGIC | DRUG-DURATION | 0.75 | 1.00 | **0.85** | 0.92 |
# MAGIC | DRUG-FORM | 0.88 | 1.00 | **0.94** | 0.95* |
# MAGIC | DRUG-FREQUENCY | 0.79 | 1.00 | **0.88** | 0.90 |
# MAGIC | DRUG-REASON | 0.60 | 1.00 | **0.75** | 0.70 |
# MAGIC | DRUG-ROUTE | 0.79 | 1.00 | **0.88** | 0.95* |
# MAGIC | DRUG-STRENGTH | 0.95 | 1.00 | **0.98** | 0.97 |
# MAGIC 
# MAGIC 
# MAGIC *Magge, Scotch, Gonzalez-Hernandez (2018) collapsed DRUG-FORM and DRUG-ROUTE into a single relation.

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("documents")

sentencer = SentenceDetector()\
    .setInputCols(["documents"])\
    .setOutputCol("sentences")

tokenizer = sparknlp.annotators.Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

posology_ner = MedicalNerModel()\
    .pretrained("ner_posology", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ners")   

posology_ner_converter = NerConverterInternal() \
    .setInputCols(["sentences", "tokens", "ners"]) \
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

reModel = RelationExtractionModel()\
    .pretrained("posology_re")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("posology_relations")\
    .setMaxSyntacticDistance(4)

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    posology_ner,
    posology_ner_converter,
    dependency_parser,
    reModel
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

results = model.transform(sparkDF)
results.printSchema()

# COMMAND ----------

results.select('posology_relations.metadata').show(5)

# COMMAND ----------

results.select('subject_id','date', F.explode(F.arrays_zip('posology_relations.result', 'posology_relations.metadata')).alias("cols")).show()

# COMMAND ----------

result_df = results.select('subject_id','date',F.explode(F.arrays_zip(results.posology_relations.result, results.posology_relations.metadata)).alias("cols")) \
                   .select('subject_id','date',F.expr("cols['0']").alias("relation"),
                                               F.expr("cols['1']['entity1']").alias("entity1"),
                                               F.expr("cols['1']['entity1_begin']").alias("entity1_begin"),
                                               F.expr("cols['1']['entity1_end']").alias("entity1_end"),
                                               F.expr("cols['1']['chunk1']").alias("chunk1"),
                                               F.expr("cols['1']['entity2']").alias("entity2"),
                                               F.expr("cols['1']['entity2_begin']").alias("entity2_begin"),
                                               F.expr("cols['1']['entity2_end']").alias("entity2_end"),
                                               F.expr("cols['1']['chunk2']").alias("chunk2"),
                                               F.expr("cols['1']['confidence']").alias("confidence"))

# COMMAND ----------

result_df.show()

# COMMAND ----------

pd_result = result_df.toPandas()
pd_result

# COMMAND ----------

outname = 'posology_re_results.csv'
outdir = f'{delta_path}/silver/dataset'
pd_result.to_csv(outdir+outname, index=False, encoding="utf-8")
display(dbutils.fs.ls(outdir))

# COMMAND ----------

temp = pd.read_csv('/dbfs/FileStore/posology_re_results.csv')
temp

# COMMAND ----------

# MAGIC %md
# MAGIC ## RxNorm Code Extraction From Re_Results

# COMMAND ----------

import pandas as pd

outname = 'posology_re_results.csv'
outdir = f'{delta_path}/silver/dataset'
pd_RE = pd.read_csv(outdir+outname, index=False, encoding="utf-8")
pd_RE

# COMMAND ----------

sp_RE = spark.createDataFrame(pd_RE)
sp_RE.show(20)

# COMMAND ----------

sp_RE.rdd.getNumPartitions()

# COMMAND ----------

# drug + strength or form
from pyspark.sql.functions import when, col

sp_RE_results = sp_RE.withColumn('rx_text',
  when( (F.col('entity1')=='DRUG') & ((F.col('entity2')=='FORM') | (F.col('entity2')=='STRENGTH') | (F.col('entity2')=='DOSAGE') ), F.concat(F.col('chunk1'),F.lit(' '), F.col('chunk2')))
 .when( ((F.col('entity1')=='FORM') | (F.col('entity1')=='STRENGTH') | (F.col('entity1')=='DOSAGE') ) & (F.col('entity2')=='DRUG'), F.concat(F.col('chunk2'),F.lit(' '), F.col('chunk1')))
 .when( (F.col('entity1')=='DRUG') & ((F.col('entity2')!='FORM') & (F.col('entity2')!='STRENGTH') & (F.col('entity2')!='DOSAGE') ), F.col('chunk1'))
 .when( (F.col('entity2')=='DRUG') & ((F.col('entity1')!='FORM') & (F.col('entity1')!='STRENGTH') & (F.col('entity1')!='DOSAGE') ), F.col('chunk2'))
                   .otherwise(F.lit(' '))
                   )

sp_RE_results.show(20,70)

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("rx_text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sentence_embeddings")
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

rxnorm_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver])

# COMMAND ----------

rxnorm_results = rxnorm_pipelineModel.transform(sp_RE_results)
rxnorm_results.printSchema(), rxnorm_results.rdd.getNumPartitions()

# COMMAND ----------

sp_rxnorm_result = rxnorm_results.select('subject_id','date', 'relation', 'entity1', 'entity1_begin','entity1_end',  'chunk1', 'entity2', 'entity2_begin', 'entity2_end', 
                                         'chunk2', 'confidence', 'rx_text', 
                                         F.explode(F.arrays_zip(rxnorm_results.ner_chunk.result, 
                                                                rxnorm_results.ner_chunk.metadata, 
                                                                rxnorm_results.rxnorm_code.result, 
                                                                rxnorm_results.rxnorm_code.metadata)).alias("cols")) \
                                     .select('subject_id','date', 'relation', 'entity1', 'entity1_begin','entity1_end',  'chunk1', 'entity2', 'entity2_begin', 'entity2_end',
                                             'chunk2', 'confidence', 'rx_text',
                                             F.expr("cols['1']['sentence']").alias("sent_id"),
                                             F.expr("cols['0']").alias("ner_chunk"),
                                             F.expr("cols['1']['entity']").alias("entity"), 
                                             F.expr("cols['2']").alias('rxnorm_code'),
                                             F.expr("cols['3']['all_k_results']").alias("all_codes"),
                                             F.expr("cols['3']['all_k_resolutions']").alias("resolutions"))

# COMMAND ----------

sp_rxnorm_result.show()

# COMMAND ----------

sp_rxnorm_result = sp_rxnorm_result.withColumn('all_codes', F.split(F.col('all_codes'), ':::'))\
                                    .withColumn('resolutions', F.split(F.col('resolutions'), ':::'))\

sp_rxnorm_result.show()

# COMMAND ----------

pd_rxnorm_result = sp_rxnorm_result.toPandas()
pd_rxnorm_result

# COMMAND ----------

outname = 'posology_RE_rxnorm_results.csv'
outdir = f'{delta_path}/silver/dataset'
pd_rxnorm_result.to_csv(outdir+outname, index=False, encoding="utf-8")
display(dbutils.fs.ls(outdir))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split Resolutions to Resolution Drug and Write Results to Golden Delta Layer

# COMMAND ----------

outname = 'posology_RE_rxnorm_results.csv'
outdir = f'{delta_path}/silver/dataset'
df = pd.read_csv(outdir+outname)
df

# COMMAND ----------

df['res']=df['resolutions'].str.split(' ').str[0]
df.res.head()

# COMMAND ----------

df['resolution'] = [val[2:] for val in df['res']]
df['resolution'].head()

# COMMAND ----------

df['drug_resolution'] = df['resolution'].str.split().str.get(0)
df['drug_resolution'] = df['drug_resolution'].replace({',':''}, regex=True)
df['drug_resolution'] = df['drug_resolution'].replace({"'":""}, regex=True)
df.head(20)

# COMMAND ----------

df['drug_resolution'] = df['drug_resolution'].str.lower()
df['chunk1'] = df['chunk1'].str.lower()
df['chunk2'] = df['chunk2'].str.lower()
df.head(20)

# COMMAND ----------

outname = 'posology_RE_rxnorm_w_drug_resolutions.csv'
outdir = f'{delta_path}/golden/dataset'
df.to_csv(outdir+outname, index=False, encoding="utf-8")
display(dbutils.fs.ls(outdir))

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER JSL Slim
# MAGIC 
# MAGIC Model card of the ner_jsl_slim is [here](https://nlp.johnsnowlabs.com/2021/08/13/ner_jsl_slim_en.html).

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")\
      .setCustomBounds(["\|"])

tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")\

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

jsl_ner = MedicalNerModel.pretrained("ner_jsl_slim", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")

jsl_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(['Symptom','Body_Part', 'Procedure', 'Disease_Syndrome_Disorder', 'Test'])

ner_pipeline = Pipeline(
    stages = [
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        jsl_ner,
        jsl_converter
        ])

data_ner = spark.createDataFrame([[""]]).toDF("text")
model = ner_pipeline.fit(data_ner)

# COMMAND ----------

results = model.transform(sparkDF)
results.printSchema()

# COMMAND ----------

result_df = results.select('subject_id','date',
                           F.explode(F.arrays_zip(results.ner_chunk.result, results.ner_chunk.begin, results.ner_chunk.end, results.ner_chunk.metadata)).alias("cols")) \
                    .select('subject_id','date',
                            F.expr("cols['3']['sentence']").alias("sentence_id"),
                            F.expr("cols['0']").alias("chunk"),
                            F.expr("cols['1']").alias("begin"),
                            F.expr("cols['2']").alias("end"),
                            F.expr("cols['3']['entity']").alias("ner_label"))\
                    .filter("ner_label!='O'")

# COMMAND ----------

result_df.show()

# COMMAND ----------

pd_result = result_df.toPandas()
pd_result

# COMMAND ----------

outname = 'ner_jsl_slim_results.csv'
outdir = f'{delta_path}/golden/dataset'
pd_result.to_csv(outdir+outname, index=False, encoding="utf-8")
display(dbutils.fs.ls(outdir))

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |BeautifulSoup|MIT License|https://www.crummy.com/software/BeautifulSoup/#Download|https://www.crummy.com/software/BeautifulSoup/bs4/download/|
# MAGIC |Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.