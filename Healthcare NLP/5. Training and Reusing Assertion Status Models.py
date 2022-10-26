# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Training and Reusing Assertion Status Models

# COMMAND ----------

import os
import json
import string
import numpy as np
import pandas as pd

import nlu
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
# MAGIC # Clinical Assertion Model (with pretrained models)

# COMMAND ----------

# MAGIC %md
# MAGIC The deep neural network architecture for assertion status detection in Spark NLP is based on a Bi-LSTM framework, and is a modified version of the architecture proposed by Federico Fancellu, Adam Lopez and Bonnie Webber ([Neural Networks For Negation Scope Detection](https://aclanthology.org/P16-1047.pdf)). Its goal is to classify the assertions made on given medical concepts as being present, absent, or possible in the patient, conditionally present in the patient under certain circumstances,
# MAGIC hypothetically present in the patient at some future point, and
# MAGIC mentioned in the patient report but associated with someoneelse.
# MAGIC In the proposed implementation, input units depend on the
# MAGIC target tokens (a named entity) and the neighboring words that
# MAGIC are explicitly encoded as a sequence using word embeddings.
# MAGIC Similar to paper mentioned above,  it is observed that that 95% of the scope tokens (neighboring words) fall in a window of 9 tokens to the left and 15
# MAGIC to the right of the target tokens in the same dataset. Therefore, the same window size was implemented and it following parameters were used: learning
# MAGIC rate 0.0012, dropout 0.05, batch size 64 and a maximum sentence length 250. The model has been implemented within
# MAGIC Spark NLP as an annotator called AssertionDLModel. After
# MAGIC training 20 epoch and measuring accuracy on the official test
# MAGIC set, this implementation exceeds the latest state-of-the-art
# MAGIC accuracy benchmarks as summarized as following table:
# MAGIC 
# MAGIC |Assertion Label|Spark NLP|Latest Best|
# MAGIC |-|-|-|
# MAGIC |Absent       |0.944 |0.937|
# MAGIC |Someone-else |0.904|0.869|
# MAGIC |Conditional  |0.441|0.422|
# MAGIC |Hypothetical |0.862|0.890|
# MAGIC |Possible     |0.680|0.630|
# MAGIC |Present      |0.953|0.957|
# MAGIC |micro F1     |0.939|0.934|

# COMMAND ----------

# MAGIC %md
# MAGIC |    | **model_name**              |**Predicted Entities**|
# MAGIC |---:|:------------------------|-|
# MAGIC |  1 | **assertion_dl**            |Present, Absent, Possible, Planned, Someoneelse, Past, Family, None, Hypotetical|
# MAGIC |  2 | **assertion_dl_biobert**    |absent, present, conditional, associated_with_someone_else, hypothetical, possible|
# MAGIC |  3 | **assertion_dl_healthcare** |absent, present, conditional, associated_with_someone_else, hypothetical, possible|
# MAGIC |  4 | **assertion_dl_large**      |hypothetical, present, absent, possible, conditional, associated_with_someone_else|
# MAGIC |  5 | **assertion_dl_radiology**  |Confirmed, Suspected, Negative|
# MAGIC |  6 | **assertion_jsl**           |Present, Absent, Possible, Planned, Someoneelse, Past, Family, None, Hypotetical|
# MAGIC |  7 | **assertion_jsl_large**     |present, absent, possible, planned, someoneelse, past|
# MAGIC |  8 | **assertion_ml**            |Hypothetical, Present, Absent, Possible, Conditional, Associated_with_someone_else|
# MAGIC |  9 | **assertion_dl_scope_L10R10**| hypothetical, associated_with_someone_else, conditional, possible, absent, present|
# MAGIC | 10 | **assertion_dl_biobert_scope_L10R10**| hypothetical, associated_with_someone_else, conditional, possible, absent, present|
# MAGIC | 11 | **assertion_jsl_augmented**| Present, Absent, Possible, Planned, Past, Family, Hypotetical, SomeoneElse|

# COMMAND ----------

# MAGIC %md
# MAGIC ### Oncology Assertion Models
# MAGIC |    | model_name              |Predicted Entities|
# MAGIC |---:|:------------------------|-|
# MAGIC | 1 | **assertion_oncology_wip** | Medical_History, Family_History, Possible, Hypothetical_Or_Absent|
# MAGIC | 2 | **assertion_oncology_problem_wip** |Present, Possible, Hypothetical, Absent, Family|
# MAGIC | 3 | **assertion_oncology_treatment_wip** |Present, Planned, Past, Hypothetical, Absent|
# MAGIC | 4 | **assertion_oncology_response_to_treatment_wip** |Present_Or_Past, Hypothetical_Or_Absent|
# MAGIC | 5 | **assertion_oncology_test_binary_wip** |Present_Or_Past, Hypothetical_Or_Absent|
# MAGIC | 6 | **assertion_oncology_smoking_status_wip** |Absent, Past, Present|
# MAGIC | 7 | **assertion_oncology_family_history_wip** |Family_History, Other|
# MAGIC | 8 | **assertion_oncology_demographic_binary_wip** |Patient, Someone_Else|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pretrained `assertion_dl` model

# COMMAND ----------

# Annotator that transforms a text column from dataframe into an Annotation ready for NLP

from sparknlp_jsl.annotator import *

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

# Sentence Detector annotator, processes various sentences per line

sentenceDetector = SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

# Tokenizer splits words in a relevant format for NLP

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")

# Clinical word embeddings trained on PubMED dataset
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

# NER model trained on i2b2 (sampled from MIMIC) dataset
clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

# Assertion model trained on i2b2 (sampled from MIMIC) dataset
# coming from sparknlp_jsl.annotator !!
clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
  .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
  .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    clinical_assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)

# COMMAND ----------

# we also have a LogReg based Assertion Model.
'''
clinical_assertion_ml = AssertionLogRegModel.pretrained("assertion_ml", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
'''

# COMMAND ----------

import pandas as pd

text = 'Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. No alopecia noted. She denies pain'

print (text)

light_model = LightPipeline(model)

light_result = light_model.fullAnnotate(text)[0]

chunks=[]
entities=[]
status=[]

for n,m in zip(light_result['ner_chunk'],light_result['assertion']):
    
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)
        
df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

df

# COMMAND ----------

import nlu
text = 'Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. No alopecia noted. She denies pain'

nlu.to_pretty_df(model,text,output_level='chunk').columns

# COMMAND ----------

cols = [
     'entities_ner_chunk',
     'entities_ner_chunk_class', 
     'assertion',]
     
df = nlu.to_pretty_df(model,text,output_level='chunk')[cols]
df


# COMMAND ----------

from sparknlp_display import AssertionVisualizer

visualizer = AssertionVisualizer()

vis = visualizer.display(light_result, 'ner_chunk', 'assertion', return_html=True)
#visualizer.set_label_colors({'TREATMENT':'#008080', 'PROBLEM':'#800080'})


displayHTML(vis)

# COMMAND ----------

! wget -q	https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/pubmed/pubmed_sample_text_small.csv -P /dbfs/

# COMMAND ----------

import pyspark.sql.functions as F

pubMedDF = spark.read\
                .option("header", "true")\
                .csv("/pubmed_sample_text_small.csv")\
                
pubMedDF.show(truncate=50)

# COMMAND ----------

result = model.transform(pubMedDF.limit(100))

# COMMAND ----------

result.show()

# COMMAND ----------

result.select('sentence.result').take(1)

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.ner_chunk.result,
                                     result.ner_chunk.begin, 
                                     result.ner_chunk.end,
                                     result.ner_chunk.metadata,
                                     result.assertion.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']").alias("begin"),
              F.expr("cols['2']").alias("end"),
              F.expr("cols['3']['entity']").alias("ner_label"),
              F.expr("cols['3']['sentence']").alias("sent_id"),
              F.expr("cols['4']").alias("assertion"),).show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Pretrained `assertion_dl_radiology` model

# COMMAND ----------

from sparknlp_jsl.annotator import *

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

# Sentence Detector annotator, processes various sentences per line
sentenceDetector = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# Tokenizer splits words in a relevant format for NLP
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

# Clinical word embeddings trained on PubMED dataset
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

# NER model for radiology
radiology_ner = MedicalNerModel.pretrained("ner_radiology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(["ImagingFindings"])

# Assertion model trained on radiology dataset
# coming from sparknlp_jsl.annotator !!

radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    radiology_ner,
    ner_converter,
    radiology_assertion
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")
radiologyAssertion_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

# A sample text from a radiology report

text = """No right-sided pleural effusion or pneumothorax is definitively seen and there are mildly displaced fractures of the left lateral 8th and likely 9th ribs."""

# COMMAND ----------

data = spark.createDataFrame([[text]]).toDF("text")

# COMMAND ----------

result = radiologyAssertion_model.transform(data)

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.ner_chunk.result, 
                                     result.ner_chunk.metadata, 
                                     result.assertion.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"),
              F.expr("cols['1']['sentence']").alias("sent_id"),
              F.expr("cols['2']").alias("assertion")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing a generic Assertion + NER function

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id


def get_base_pipeline (embeddings = 'embeddings_clinical'):

    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

  # Sentence Detector annotator, processes various sentences per line
    sentenceDetector = SentenceDetector()\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

  # Tokenizer splits words in a relevant format for NLP
    tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

  # Clinical word embeddings trained on PubMED dataset
    word_embeddings = WordEmbeddingsModel.pretrained(embeddings, "en", "clinical/models")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("embeddings")

    base_pipeline = Pipeline(stages=[
                        documentAssembler,
                        sentenceDetector,
                        tokenizer,
                        word_embeddings])

    return base_pipeline



def get_clinical_assertion (embeddings, spark_df, nrows = 100, model_name = 'ner_clinical'):

  # NER model trained on i2b2 (sampled from MIMIC) dataset
    loaded_ner_model = MedicalNerModel.pretrained(model_name, "en", "clinical/models") \
        .setInputCols(["sentence", "token", "embeddings"]) \
        .setOutputCol("ner")

    ner_converter = NerConverter() \
        .setInputCols(["sentence", "token", "ner"]) \
        .setOutputCol("ner_chunk")

  # Assertion model trained on i2b2 (sampled from MIMIC) dataset
  # coming from sparknlp_jsl.annotator !!
    clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
        .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
        .setOutputCol("assertion")
      

    base_model = get_base_pipeline (embeddings)

    nlpPipeline = Pipeline(stages=[
        base_model,
        loaded_ner_model,
        ner_converter,
        clinical_assertion])

    empty_data = spark.createDataFrame([[""]]).toDF("text")

    model = nlpPipeline.fit(empty_data)

    result = model.transform(spark_df.limit(nrows))

    result = result.withColumn("id", monotonically_increasing_id())

    result_df = result.select(F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata, result.assertion.result)).alias("cols")) \
                      .select(F.expr("cols['0']").alias("chunk"),
                              F.expr("cols['1']['entity']").alias("ner_label"),
                              F.expr("cols['2']").alias("assertion"))\
                              .filter("ner_label!='O'")

    return result_df

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_clinical_large'

nrows = 1000

ner_df = get_clinical_assertion (embeddings, pubMedDF, nrows, model_name)

ner_df.show()

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_posology'

nrows = 100

ner_df = get_clinical_assertion (embeddings, pubMedDF, nrows, model_name)

ner_df.show()

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_posology_greedy'

entry_data = spark.createDataFrame([["The patient did not take a capsule of Advil"]]).toDF("text")

ner_df = get_clinical_assertion (embeddings, entry_data, nrows, model_name)

ner_df.show()

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_clinical'

entry_data = spark.createDataFrame([["The patient has no fever"]]).toDF("text")

ner_df = get_clinical_assertion (embeddings, entry_data, nrows, model_name)

ner_df.show()

# COMMAND ----------

import pandas as pd

def get_clinical_assertion_light (light_model, text):

  light_result = light_model.fullAnnotate(text)[0]

  chunks=[]
  entities=[]
  status=[]

  for n,m in zip(light_result['ner_chunk'],light_result['assertion']):
      
      chunks.append(n.result)
      entities.append(n.metadata['entity']) 
      status.append(m.result)
          
  df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

  return df

# COMMAND ----------

clinical_text = """
Patient with severe fever and sore throat. 
He shows no stomach pain and he maintained on an epidural and PCA for pain control.
He also became short of breath with climbing a flight of stairs.
After CT, lung tumor located at the right lower lobe. Father with Alzheimer.
"""

light_model = LightPipeline(model)

get_clinical_assertion_light (light_model, clinical_text)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Assertion with BioNLP (Cancer Genetics) NER

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_bionlp'

nrows = 100

ner_df = get_clinical_assertion (embeddings, pubMedDF, nrows, model_name)

ner_df.show(truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Assertion Filterer
# MAGIC AssertionFilterer will allow you to filter out the named entities by the list of acceptable assertion statuses. This annotator would be quite handy if you want to set a white list for the acceptable assertion statuses like present or conditional; and do not want absent conditions get out of your pipeline.

# COMMAND ----------

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")

ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
      .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
      .setOutputCol("assertion")

assertion_filterer = AssertionFilterer()\
      .setInputCols("sentence","ner_chunk","assertion")\
      .setOutputCol("assertion_filtered")\
      .setWhiteList(["present"])

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter,
      clinical_assertion,
      assertion_filterer
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")
assertionFilter_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

text = 'Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. Alopecia noted. She denies pain.'

light_model = LightPipeline(assertionFilter_model)
light_result = light_model.annotate(text)

light_result.keys()

# COMMAND ----------

list(zip(light_result['ner_chunk'], light_result['assertion']))

# COMMAND ----------

assertion_filterer.getWhiteList()

# COMMAND ----------

light_result['assertion_filtered']

# COMMAND ----------

# MAGIC %md
# MAGIC # Train a custom Assertion Model

# COMMAND ----------

# MAGIC %md
# MAGIC **WARNING:** For training an Assertion model, please use TensorFlow version 2.3

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/i2b2_assertion_sample_short.csv -P /dbfs/

# COMMAND ----------

assertion_df = spark.read.option("header", True).option("inferSchema", "True").csv("/i2b2_assertion_sample_short.csv")


# COMMAND ----------

assertion_df.show(3, truncate=100)

# COMMAND ----------

(training_data, test_data) = assertion_df.randomSplit([0.8, 0.2], seed = 100)
print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))

# COMMAND ----------

training_data.groupBy('label').count().orderBy('count', ascending=False).show(truncate=False)


# COMMAND ----------

document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

chunk = Doc2Chunk()\
    .setInputCols("document")\
    .setOutputCol("chunk")\
    .setChunkCol("target")\
    .setStartCol("start")\
    .setStartColByTokenIndex(True)\
    .setFailOnMissing(False)\
    .setLowerCase(True)

token = Tokenizer()\
    .setInputCols(['document'])\
    .setOutputCol('token')

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["document", "token"])\
      .setOutputCol("embeddings")


# COMMAND ----------

# MAGIC %md
# MAGIC We will transform our test data with a pipeline consisting of same steps with the pipeline which contains AssertionDLApproach.
# MAGIC By doing this, we enable that test data will have same columns with training data in AssertionDLApproach. <br/>
# MAGIC The goal of this implementation is enabling the usage of `setTestDataset()` parameter in AssertionDLApproach.

# COMMAND ----------

clinical_assertion_pipeline = Pipeline(
    stages = [
    document,
    chunk,
    token,
    embeddings])

assertion_test_data = clinical_assertion_pipeline.fit(test_data).transform(test_data)

# COMMAND ----------

assertion_test_data.columns

# COMMAND ----------

# MAGIC %md
# MAGIC We save the test data in parquet format to use in `AssertionDLApproach()`.

# COMMAND ----------

assertion_test_data.write.parquet('i2b2_assertion_sample_test_data.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graph Setup

# COMMAND ----------

# !pip install -q tensorflow==2.7.0
# !pip install -q tensorflow-addons

# COMMAND ----------

from sparknlp_jsl.annotator import TFGraphBuilder

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/tf_graphs

# COMMAND ----------

assertion_graph_builder = TFGraphBuilder()\
    .setModelName("assertion_dl")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label")\
    .setGraphFolder('file:/dbfs/tf_graphs')\
    .setGraphFile("assertion_graph.pb")\
    .setMaxSequenceLength(250)\
    .setHiddenUnitsNumber(25)

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/assertion_logs

# COMMAND ----------

 # %fs mkdirs file:/dbfs/assertion_tf_graphs
 # %fs mkdirs file:/dbfs/assertion_logs

# if you want you can use existing graph

# !wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/tf_graphs/blstm_34_32_30_200_2.pb -P /dbfs/assertion_tf_graphs

# COMMAND ----------

# Create custom graph


# tf_graph.print_model_params("assertion_dl")

# feat_size = 200
# n_classes = 2

# tf_graph.build("assertion_dl",
#                build_params={"n_classes": n_classes},
#                model_location= "/dbfs/assertion_tf_graphs", 
#                model_filename="blstm_34_32_30_{}_{}.pb".format(feat_size, n_classes))

# COMMAND ----------

# MAGIC %md
# MAGIC **Setting the Scope Window (Target Area) Dynamically in Assertion Status Detection Models**
# MAGIC 
# MAGIC 
# MAGIC This parameter allows you to train the Assertion Status Models to focus on specific context windows when resolving the status of a NER chunk. The window is in format `[X,Y]` being `X` the number of tokens to consider on the left of the chunk, and `Y` the max number of tokens to consider on the right. Let’s take a look at what different windows mean:
# MAGIC 
# MAGIC 
# MAGIC *   By default, the window is `[-1,-1]` which means that the Assertion Status will look at all of the tokens in the sentence/document (up to a maximum of tokens set in `setMaxSentLen()` ).
# MAGIC *   `[0,0]` means “don’t pay attention to any token except the ner_chunk”, what basically is not considering any context for the Assertion resolution.
# MAGIC *   `[9,15]` is what empirically seems to be the best baseline, meaning that we look up to 9 tokens on the left and 15 on the right of the ner chunk to understand the context and resolve the status.
# MAGIC 
# MAGIC 
# MAGIC Check this [Scope Window Tuning Assertion Status Detection notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.1.Scope_window_tuning_assertion_status_detection.ipynb)  that illustrates the effect of the different windows and how to properly fine-tune your AssertionDLModels to get the best of them.
# MAGIC 
# MAGIC In our case, the best Scope Window is around [10,10]

# COMMAND ----------

scope_window = [10,10]

assertionStatus = AssertionDLApproach()\
    .setLabelCol("label")\
    .setInputCols("document", "chunk", "embeddings")\
    .setOutputCol("assertion")\
    .setBatchSize(128)\
    .setDropout(0.1)\
    .setLearningRate(0.001)\
    .setEpochs(50)\
    .setValidationSplit(0.2)\
    .setStartCol("start")\
    .setEndCol("end")\
    .setMaxSentLen(250)\
    .setEnableOutputLogs(True)\
    .setOutputLogsPath('dbfs:/assertion_logs')\
    .setGraphFolder('dbfs:/tf_graphs')\
    .setGraphFile("file:/dbfs/tf_graphs/assertion_graph.pb")\
    .setTestDataset(path="/i2b2_assertion_sample_test_data.parquet", read_as='SPARK', options={'format': 'parquet'})\
    .setScopeWindow(scope_window)


'''
If .setTestDataset parameter is employed, raw test data cannot be fitted. .setTestDataset only works for dataframes which are correctly transformed
by a pipeline consisting of document, chunk, embeddings stages.
'''

# COMMAND ----------

clinical_assertion_pipeline = Pipeline(
    stages = [
    document,
    chunk,
    token,
    embeddings,
    assertion_graph_builder,
    assertionStatus])

# COMMAND ----------

assertion_model = clinical_assertion_pipeline.fit(training_data)

# COMMAND ----------

assertion_model.stages

# COMMAND ----------

preds = assertion_model.transform(test_data).select('label','assertion.result')

preds.show()

# COMMAND ----------

preds_df = preds.toPandas()

# COMMAND ----------

preds_df['result'] = preds_df['result'].apply(lambda x : x[0])


# COMMAND ----------

# We are going to use sklearn to evalute the results on test dataset
from sklearn.metrics import classification_report

print (classification_report(preds_df['result'], preds_df['label']))

# COMMAND ----------

#saving the model that we've trained
assertion_model.stages[-1].write().overwrite().save('/databricks/driver/models/custom_assertion_model')