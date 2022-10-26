# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clinical Deidentification

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

print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC # Deidentification Model

# COMMAND ----------

# MAGIC %md
# MAGIC Protected Health Information: 
# MAGIC - individual’s past, present, or future physical or mental health or condition
# MAGIC - provision of health care to the individual
# MAGIC - past, present, or future payment for the health care 
# MAGIC 
# MAGIC Protected health information includes many common identifiers (e.g., name, address, birth date, Social Security Number) when they can be associated with the health information.

# COMMAND ----------

# MAGIC %md
# MAGIC <center><b>Deidentification NER Models for English</b></center>
# MAGIC 
# MAGIC |index|model|lang|index|model|lang|index|model|lang|
# MAGIC |-----:|:-----|----|-----:|:-----|----|-----:|:-----|----|
# MAGIC | 1| [deidentify_dl](https://nlp.johnsnowlabs.com/2021/01/28/deidentify_dl_en.html)  |en| 7| [ner_deid_enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_enriched_biobert_en.html)  |en| 13| [ner_deid_subentity_augmented](https://nlp.johnsnowlabs.com/2021/09/03/ner_deid_subentity_augmented_en.html)  |en|
# MAGIC | 2| [deidentify_large](https://nlp.johnsnowlabs.com/2020/08/04/deidentify_large_en.html)  |en| 8| [ner_deid_generic_augmented](https://nlp.johnsnowlabs.com/2021/06/30/ner_deid_generic_augmented_en.html)  |en| 14| [ner_deid_subentity_augmented_i2b2](https://nlp.johnsnowlabs.com/2021/11/29/ner_deid_subentity_augmented_i2b2_en.html)  |en|
# MAGIC | 3| [deidentify_rb](https://nlp.johnsnowlabs.com/2019/06/04/deidentify_rb_en.html)  |en| 9| [ner_deid_generic_glove](https://nlp.johnsnowlabs.com/2021/06/06/ner_deid_generic_glove_en.html)  |en| 15| [ner_deid_subentity_glove](https://nlp.johnsnowlabs.com/2021/06/06/ner_deid_subentity_glove_en.html)  |en|
# MAGIC | 4| [ner_deid_augmented](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_augmented_en.html)  |en| 10| [ner_deid_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_large_en.html)  |en| 16| [ner_deid_synthetic](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_synthetic_en.html)  |en|
# MAGIC | 5| [ner_deid_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_biobert_en.html)  |en| 11| [ner_deid_sd](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_en.html)  |en| 17| [ner_deidentify_dl](https://nlp.johnsnowlabs.com/2021/03/31/ner_deidentify_dl_en.html)  |en|
# MAGIC | 6| [ner_deid_enriched](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_enriched_en.html)  |en| 12| [ner_deid_sd_large](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_large_en.html)  |en|

# COMMAND ----------

# MAGIC %md
# MAGIC <center><b>Deidentification NER Models for Other Languages</b></center>
# MAGIC 
# MAGIC |index|model|lang|index|model|lang|
# MAGIC |-----:|:-----|----|-----:|:-----|----|
# MAGIC | 1| [ner_deid_generic](https://nlp.johnsnowlabs.com/2022/01/06/ner_deid_generic_de.html)  |de| 11| [ner_deid_generic](https://nlp.johnsnowlabs.com/2022/02/11/ner_deid_generic_fr.html)  |fr|
# MAGIC | 2| [ner_deid_subentity](https://nlp.johnsnowlabs.com/2022/01/06/ner_deid_subentity_de.html)  |de| 12| [ner_deid_subentity](https://nlp.johnsnowlabs.com/2022/02/14/ner_deid_subentity_fr.html)  |fr|
# MAGIC | 3| [ner_deid_generic](https://nlp.johnsnowlabs.com/2022/01/18/ner_deid_generic_es.html)  |es| 13| [ner_deid_generic](https://nlp.johnsnowlabs.com/2022/03/25/ner_deid_generic_it_3_0.html)  |it|
# MAGIC | 4| [ner_deid_generic_augmented](https://nlp.johnsnowlabs.com/2022/02/16/ner_deid_generic_augmented_es.html)  |es| 14| [ner_deid_subentity](https://nlp.johnsnowlabs.com/2022/03/25/ner_deid_subentity_it_2_4.html)  |it|
# MAGIC | 5| [ner_deid_generic_roberta](https://nlp.johnsnowlabs.com/2022/01/17/ner_deid_generic_roberta_es.html)  |es| 15| [ner_deid_generic](https://nlp.johnsnowlabs.com/2022/04/13/ner_deid_generic_pt_3_0.html)  |pt|
# MAGIC | 6| [ner_deid_generic_roberta_augmented](https://nlp.johnsnowlabs.com/2022/02/16/ner_deid_generic_roberta_augmented_es.html)  |es| 16| [ner_deid_subentity](https://nlp.johnsnowlabs.com/2022/04/13/ner_deid_subentity_pt_3_0.html)  |pt|
# MAGIC | 7| [ner_deid_subentity](https://nlp.johnsnowlabs.com/2022/01/18/ner_deid_subentity_es.html)  |es| 17| [ner_deid_subentity](https://nlp.johnsnowlabs.com/2022/06/27/ner_deid_subentity_ro_3_0.html)  |ro|
# MAGIC | 8| [ner_deid_subentity_augmented](https://nlp.johnsnowlabs.com/2022/02/16/ner_deid_subentity_augmented_es.html)  |es| 18| [ner_deid_subentity_bert](https://nlp.johnsnowlabs.com/2022/06/27/ner_deid_subentity_bert_ro_3_0.html)  |ro|
# MAGIC | 9| [ner_deid_subentity_roberta](https://nlp.johnsnowlabs.com/2022/01/17/ner_deid_subentity_roberta_es.html)  |es| 19| [ner_deid_generic](https://nlp.johnsnowlabs.com/models)  |ro|
# MAGIC | 10| [ner_deid_subentity_roberta_augmented](https://nlp.johnsnowlabs.com/2022/02/16/ner_deid_subentity_roberta_augmented_es.html)  |es| 20| [ner_deid_generic_bert](https://nlp.johnsnowlabs.com/models)  |ro|

# COMMAND ----------

# MAGIC %md
# MAGIC You can find German, Spanish, French, Italian, Portuguese and Romanian deidentification models and pretrained pipeline examples in this notebook:   [Clinical Multi Language Deidentification Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.1.Clinical_Multi_Language_Deidentification.ipynb)

# COMMAND ----------

# MAGIC %md
# MAGIC **Let's start!**
# MAGIC 
# MAGIC Load NER pipeline to identify protected entities:

# COMMAND ----------

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

# NER model trained on n2c2 (de-identification and Heart Disease Risk Factors Challenge) datasets)

clinical_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pretrained NER models extracts:
# MAGIC 
# MAGIC - Name
# MAGIC - Profession
# MAGIC - Age
# MAGIC - Date
# MAGIC - Contact(Telephone numbers, FAX numbers, Email addresses)
# MAGIC - Location (Address, City, Postal code, Hospital Name, Employment information)
# MAGIC - Id (Social Security numbers, Medical record numbers, Internet protocol addresses)

# COMMAND ----------

text ='''
Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson Ora , MR # 7194334 Date : 01/13/93 . PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital , 0295 Keats Street , Phone 55-555-5555 .
'''

# COMMAND ----------

result = model.transform(spark.createDataFrame([[text]]).toDF("text"))

# COMMAND ----------

result_df = result.select(F.explode(F.arrays_zip(result.token.result, 
                                                 result.ner.result)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("ner_label"))

# COMMAND ----------

result_df.select("token", "ner_label").groupBy('ner_label').count().orderBy('count', ascending=False).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check extracted sensitive entities

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.ner_chunk.result,
                                     result.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Excluding entities from deidentification

# COMMAND ----------

# MAGIC %md
# MAGIC Sometimes we need to leave some entities in the text, for example, if we want to analyze the frequency of the disease by the hospital. In this case, we need to use parameter **`setWhiteList()`** to modify `ner_chunk` output. This parameter having using a list of entities type to deidentify as an input. So, if we want to leave the location in the list we need to remove this tag from the list:

# COMMAND ----------

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk") \
    .setWhiteList(['NAME', 'PROFESSION', 'ID', 'AGE', 'DATE'])

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model_with_white_list = nlpPipeline.fit(empty_data)

# COMMAND ----------

result_with_white_list = model_with_white_list.transform(spark.createDataFrame([[text]]).toDF("text"))

# COMMAND ----------

print("All Labels :")
result.select(F.explode(F.arrays_zip(result.ner_chunk.result, 
                                     result.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

print("WhiteListed Labels: ")
result_with_white_list.select(F.explode(F.arrays_zip(result_with_white_list.ner_chunk.result, 
                                                     result_with_white_list.ner_chunk.metadata)).alias("cols")) \
                      .select(F.expr("cols['0']").alias("chunk"),
                              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Masking and Obfuscation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Replace these entities with Tags

# COMMAND ----------

ner_converter = NerConverterInternal()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk") 

deidentification = DeIdentification() \
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("deidentified") \
      .setMode("mask")\
      .setReturnEntityMappings(True) #  return a new column to save the mappings between the mask/obfuscated entities and original entities.
      #.setMappingsColumn("MappingCol") # change the name of the column, 'aux' is default

deidPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter,
      deidentification])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model_deid = deidPipeline.fit(empty_data)

# COMMAND ----------

result = model_deid.transform(spark.createDataFrame([[text]]).toDF("text"))

# COMMAND ----------

result.show()

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.sentence.result, 
                                     result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), 
              F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC We have three modes to mask the entities in the Deidentification annotator. You can select the modes using the `.setMaskingPolicy()` parameter. The methods are the followings:
# MAGIC 
# MAGIC **“entity_labels”**: Mask with the entity type of that chunk. (default) <br/>
# MAGIC **“same_length_chars”**: Mask the deid entities with same length of asterix ( * ) with brackets ( [ , ] ) on both end. <br/>
# MAGIC **“fixed_length_chars”**: Mask the deid entities with a fixed length of asterix ( * ). The length is setting up using the `setFixedMaskLength()` method. <br/>
# MAGIC Let's try each of these and compare the results.

# COMMAND ----------

#deid model with "entity_labels"
deid_entity_labels= DeIdentification()\
    .setInputCols(["sentence", "token", "ner_chunk"])\
    .setOutputCol("deid_entity_label")\
    .setMode("mask")\
    .setReturnEntityMappings(True)\
    .setMaskingPolicy("entity_labels")

#deid model with "same_length_chars"
deid_same_length= DeIdentification()\
    .setInputCols(["sentence", "token", "ner_chunk"])\
    .setOutputCol("deid_same_length")\
    .setMode("mask")\
    .setReturnEntityMappings(True)\
    .setMaskingPolicy("same_length_chars")

#deid model with "fixed_length_chars"
deid_fixed_length= DeIdentification()\
    .setInputCols(["sentence", "token", "ner_chunk"])\
    .setOutputCol("deid_fixed_length")\
    .setMode("mask")\
    .setReturnEntityMappings(True)\
    .setMaskingPolicy("fixed_length_chars")\
    .setFixedMaskLength(4)


deidPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter,
      deid_entity_labels,
      deid_same_length,
      deid_fixed_length])


empty_data = spark.createDataFrame([[""]]).toDF("text")
model_deid = deidPipeline.fit(empty_data)

# COMMAND ----------

policy_result = model_deid.transform(spark.createDataFrame([[text]]).toDF("text"))

# COMMAND ----------

policy_result.show()

# COMMAND ----------

policy_result.select(F.explode(F.arrays_zip(policy_result.sentence.result, 
                                            policy_result.deid_entity_label.result, 
                                            policy_result.deid_same_length.result, 
                                            policy_result.deid_fixed_length.result)).alias("cols"))\
             .select(F.expr("cols['0']").alias("sentence"),
                     F.expr("cols['1']").alias("deid_entity_label"),
                     F.expr("cols['2']").alias("deid_same_length"),
                     F.expr("cols['3']").alias("deid_fixed_length")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mapping Column

# COMMAND ----------

result.select("aux").show(truncate=False)

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.aux.metadata, result.aux.result, result.aux.begin, result.aux.end)).alias("cols")) \
      .select(F.expr("cols['0']['originalChunk']").alias("chunk"),
              F.expr("cols['0']['beginOriginalChunk']").alias("beginChunk"),
              F.expr("cols['0']['endOriginalChunk']").alias("endChunk"),
              F.expr("cols['1']").alias("label"),
              F.expr("cols['2']").alias("beginLabel"),
              F.expr("cols['3']").alias("endLabel"),
              ).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reidentification
# MAGIC 
# MAGIC We can use `ReIdentification` annotator to go back to the original sentence.

# COMMAND ----------

reIdentification = ReIdentification()\
     .setInputCols(["aux","deidentified"])\
     .setOutputCol("original")

# COMMAND ----------

reid_result = reIdentification.transform(result)

# COMMAND ----------

reid_result.show()

# COMMAND ----------

print(text)

reid_result.select('original.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using multiple NER in the same pipeline

# COMMAND ----------

from sparknlp_jsl.annotator import *

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

deid_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_generic")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner_generic"])\
    .setOutputCol("ner_generic_chunk")\
    .setWhiteList(['ID', 'DATE', 'AGE', 'NAME', 'PROFESSION'])# CONTACT and LOCATION is removed

deid_ner_enriched = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_subentity")

ner_converter_enriched = NerConverter() \
    .setInputCols(["sentence", "token", "ner_subentity"]) \
    .setOutputCol("ner_subentity_chunk")\
    .setWhiteList(['COUNTRY', 'CITY', 'HOSPITAL', 'STATE', 'STREET', 'ZIP'])
    # we can also add PATIENT and DOCTOR entities and remove NAME entity from the other NER model

chunk_merge = ChunkMergeApproach()\
    .setInputCols("ner_subentity_chunk","ner_generic_chunk")\
    .setOutputCol("deid_merged_chunk")

deidentification = DeIdentification() \
    .setInputCols(["sentence", "token", "deid_merged_chunk"]) \
    .setOutputCol("deidentified") \
    .setMode("mask")\
    .setIgnoreRegex(True)


nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      deid_ner,
      ner_converter,
      deid_ner_enriched,
      ner_converter_enriched,
      chunk_merge,
      deidentification])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

# COMMAND ----------

text ='''
Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson Ora , MR # 7194334 Date : 01/13/93 . PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital , 0295 Keats Street , Phone 302-786-5227.
'''

# COMMAND ----------

result = model.transform(spark.createDataFrame([[text]]).toDF("text"))

# ner_deid_generic_augmented
result.select(F.explode(F.arrays_zip(result.ner_generic_chunk.result, result.ner_generic_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

# ner_deid_subentity_augmented
result.select(F.explode(F.arrays_zip(result.ner_subentity_chunk.result, result.ner_subentity_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

# merged chunk
result.select(F.explode(F.arrays_zip(result.deid_merged_chunk.result, result.deid_merged_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.sentence.result, 
                                     result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), 
              F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enriching with Regex and Override NER

# COMMAND ----------

# Text with MR number
text ='''Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street.'''

# COMMAND ----------

deidentification = DeIdentification()\
    .setInputCols(["sentence", "token", "deid_merged_chunk"])\
    .setOutputCol("deidentified")\
    .setMode("mask") \
    .setRegexOverride(False) # Prioritizing NER model

pipeline = Pipeline(stages=[
    nlpPipeline, 
    deidentification
])

model_default_rgx = pipeline.fit(empty_data)

# COMMAND ----------

result = model_default_rgx.transform(spark.createDataFrame([[text]]).toDF("text"))

result.select(F.explode(F.arrays_zip(result.sentence.result, 
                                     result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

# Creating regex rule for detecting MR number and AGE
rgx = '''NUMBER ([0-9]{2})
NUMBER (\d{7})''' 

with open("/dbfs/databricks/driver/custom_regex.txt", "w") as f:
  f.write(rgx)

f = open("/dbfs/databricks/driver/custom_regex.txt", "r")

print(f.read())

# COMMAND ----------

# MAGIC %md
# MAGIC We see that two entities  have conflict between the regex and the NER. NER has the priroty as a default. We can change this `setRegexOverride` param

# COMMAND ----------

deidentification_rgx = DeIdentification()\
    .setInputCols(["sentence", "token", "deid_merged_chunk"])\
    .setOutputCol("deidentified")\
    .setMode("mask") \
    .setRegexPatternsDictionary("dbfs:/databricks/driver/custom_regex.txt")\
    .setRegexOverride(True) # Prioritizing regex rules

nlpPipeline_rgx = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    deid_ner,
    ner_converter,
    deid_ner_enriched,
    ner_converter_enriched,
    chunk_merge,
    deidentification_rgx])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model_rgx = nlpPipeline_rgx.fit(empty_data)

# COMMAND ----------

result = model_rgx.transform(spark.createDataFrame([[text]]).toDF("text"))

result.select(F.explode(F.arrays_zip(result.sentence.result, 
                                     result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC `.setBlackList()` parameter so that not deidentifiy the labels that are specified in the list. This parameter filters just the detected Regex Entities.

# COMMAND ----------

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setWhiteList(["NAME", "LOCATION"])

# DATE, PHONE, URL, EMAIL, ZIP, DATE, SSN, PASSPORT, DLN, NPI, C_CARD, EMAIL, IBAN, DEA
deidentification = DeIdentification() \
    .setInputCols(["sentence", "token", "ner_chunk"]) \
    .setOutputCol("deidentified") \
    .setMode("mask")\
    .setRegexOverride(True)\
    .setBlackList(["DATE", "PHONE"]) # List of entities ignored for masking or obfuscation, default listed above

deidPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    deidentification])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model_deid = deidPipeline.fit(empty_data)

# COMMAND ----------

text ='''
Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street. Phone (302) 786-5227.
'''
result = model_deid.transform(spark.createDataFrame([[text]]).toDF("text"))

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.sentence.result, 
                                     result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obfuscation mode

# COMMAND ----------

# MAGIC %md
# MAGIC In the obfuscation mode **DeIdentificationModel** will replace sensetive entities with random values of the same type.

# COMMAND ----------

obs_lines = """Marvin MARSHALL#PATIENT
Hubert GROGAN#PATIENT
ALTHEA COLBURN#PATIENT
Kalil AMIN#PATIENT
Inci FOUNTAIN#PATIENT
Ekaterina Rosa#DOCTOR
Rudiger Chao#DOCTOR
COLLETTE KOHLER#DOCTOR
Mufi HIGGS#DOCTOR"""


with open ('/dbfs/databricks/driver/obfuscation.txt', 'w') as f:
  f.write(obs_lines)

# COMMAND ----------

deid_ner_enriched = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_subentity")

ner_converter_enriched = NerConverter() \
    .setInputCols(["sentence", "token", "ner_subentity"]) \
    .setOutputCol("ner_subentity_chunk")\

obfuscation = DeIdentification()\
    .setInputCols(["sentence", "token", "ner_subentity_chunk"]) \
    .setOutputCol("deidentified") \
    .setMode("obfuscate")\
    .setObfuscateDate(True)\
    #.setObfuscateRefFile('obfuscation.txt')\
    #.setObfuscateRefSource("both") #default: "faker"

pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    deid_ner_enriched,
    ner_converter_enriched,
    obfuscation
])

obfuscation_model = pipeline.fit(empty_data)

# COMMAND ----------

text ='''
Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR # 7194334 Date : 01/13/93 . Patient : Oliveira, 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street
'''

result = obfuscation_model.transform(spark.createDataFrame([[text]]).toDF("text"))

result.select(F.explode(F.arrays_zip(result.sentence.result, result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

result.select("ner_subentity_chunk").collect()

# COMMAND ----------

obfuscation = DeIdentification()\
    .setInputCols(["sentence", "token", "ner_subentity_chunk"]) \
    .setOutputCol("deidentified") \
    .setMode("obfuscate")\
    .setObfuscateDate(True)\
    .setObfuscateRefFile('/databricks/driver/obfuscation.txt')\
    .setObfuscateRefSource("file")

pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    deid_ner_enriched,
    ner_converter_enriched,
    obfuscation
])

obfuscation_model = pipeline.fit(empty_data)      
      
      
result = obfuscation_model.transform(spark.createDataFrame([[text]]).toDF("text"))

result.select(F.explode(F.arrays_zip(result.sentence.result, 
                                     result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), 
              F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Faker mode

# COMMAND ----------

# MAGIC %md
# MAGIC The faker module allow to the user to use a set of fake entities that are in the memory of the spark-nlp-internal. You can setting up this module using the the following property setObfuscateRefSource('faker').
# MAGIC 
# MAGIC If we select the setObfuscateRefSource('both') then we choose randomly the entities using the faker and the fakes entities from the obfuscateRefFile.
# MAGIC 
# MAGIC 
# MAGIC The entities that are allowed right now are the followings:
# MAGIC 
# MAGIC * Location
# MAGIC * Location-other
# MAGIC * Hospital
# MAGIC * City
# MAGIC * State
# MAGIC * Zip
# MAGIC * Country
# MAGIC * Contact
# MAGIC * Username
# MAGIC * Phone
# MAGIC * Fax
# MAGIC * Url
# MAGIC * Email
# MAGIC * Profession
# MAGIC * Name
# MAGIC * Doctor
# MAGIC * Patient
# MAGIC * Id
# MAGIC * Idnum
# MAGIC * Bioid
# MAGIC * Age
# MAGIC * Organization
# MAGIC * Healthplan
# MAGIC * Medicalrecord
# MAGIC * Ssn
# MAGIC * Passport
# MAGIC * DLN
# MAGIC * NPI
# MAGIC * C_card
# MAGIC * IBAN
# MAGIC * DEA
# MAGIC * Device

# COMMAND ----------

obfuscation = DeIdentification()\
    .setInputCols(["sentence", "token", "ner_subentity_chunk"]) \
    .setOutputCol("deidentified") \
    .setMode("obfuscate")\
    .setObfuscateDate(True)\
    .setObfuscateRefSource("faker") \

pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    deid_ner_enriched,
    ner_converter_enriched,
    obfuscation
])

obfuscation_model = pipeline.fit(empty_data)

# COMMAND ----------

text ='''
Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR # 7194334 Date : 01/13/93 . Patient : Oliveira, 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street
'''

result = obfuscation_model.transform(spark.createDataFrame([[text]]).toDF("text"))

result.select(F.explode(F.arrays_zip(result.sentence.result, 
                                     result.deidentified.result)).alias("cols")) \
      .select(F.expr("cols['0']").alias("sentence"), F.expr("cols['1']").alias("deidentified")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use full pipeline in the Light model

# COMMAND ----------

light_model = LightPipeline(model)
annotated_text = light_model.annotate(text)
annotated_text['deidentified']

# COMMAND ----------

obf_light_model = LightPipeline(obfuscation_model)
annotated_text = obf_light_model.annotate(text)
annotated_text['deidentified']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shifting Days
# MAGIC 
# MAGIC In the examples above, we replaced date entities with another date randomly. Instead of that, we can shift the date according to any column.

# COMMAND ----------

data = pd.DataFrame(
    {'patientID' : ['A001', 'A001', 
                    'A003', 'A003'],
     'text' : ['Chris Brown was discharged on 10/02/2022', 
               'Mark White was discharged on 10/04/2022', 
               'John was discharged on 15/03/2022',
               'John Moore was discharged on 15/12/2022'
              ]
    }
)

my_input_df = spark.createDataFrame(data)

my_input_df.show(truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shifting days according to the ID column
# MAGIC 
# MAGIC We use the `DocumentHashCoder()` annotator to determine shifting days. This annotator gets the hash of the specified column and creates a new document column containing day shift information. And then, the `DeIdentification` annotator deidentifies this new doc. We should set the seed parameter to hash consistently.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

documentHasher = DocumentHashCoder()\
    .setInputCols("document")\
    .setOutputCol("document2")\
    .setPatientIdColumn("patientID")\
    .setRangeDays(100)\
    .setNewDateShift("shift_days")\
    .setSeed(100)

tokenizer = Tokenizer()\
    .setInputCols(["document2"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["document2", "token"])\
    .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel\
    .pretrained("ner_deid_subentity_augmented", "en", "clinical/models")\
    .setInputCols(["document2","token", "word_embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["document2", "token", "ner"])\
    .setOutputCol("ner_chunk")

de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "document2"]) \
    .setOutputCol("deid_text") \
    .setMode("obfuscate") \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setLanguage("en") \
    .setObfuscateRefSource('faker') \
    .setUseShifDays(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    documentHasher,
    tokenizer,
    embeddings,
    clinical_ner,
    ner_converter,
    de_identification
    
])

empty_data = spark.createDataFrame([["", ""]]).toDF("text", "patientID")

pipeline_model = pipeline.fit(empty_data)

# COMMAND ----------

output = pipeline_model.transform(my_input_df)

output.select('patientID','text', 'deid_text.result').show(truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that dates are shifted 20 days for patient "A001".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shifting days according to specified values
# MAGIC 
# MAGIC Instead of shifting days according to ID column, we can specify shifting values with another column.
# MAGIC 
# MAGIC ```python
# MAGIC documentHasher = DocumentHashCoder()\
# MAGIC     .setInputCols("document")\
# MAGIC     .setOutputCol("document2")\
# MAGIC     .setDateShiftColumn("dateshift")\
# MAGIC ```

# COMMAND ----------

data = pd.DataFrame(
    {'patientID' : ['A001', 'A001', 
                    'A003', 'A003'],
     'text' : ['Chris Brown was discharged on 10/02/2022', 
               'Mark White was discharged on 10/04/2022', 
               'John was discharged on 15/03/2022',
               'John Moore was discharged on 15/12/2022'
              ],
     'dateshift' : ['10', '10', 
                    '30', '30']
    }
)

my_input_df = spark.createDataFrame(data)

my_input_df.show(truncate=False)

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

documentHasher = DocumentHashCoder()\
    .setInputCols("document")\
    .setOutputCol("document2")\
    .setDateShiftColumn("dateshift")\

tokenizer = Tokenizer()\
    .setInputCols(["document2"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["document2", "token"])\
    .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel\
    .pretrained("ner_deid_subentity_augmented", "en", "clinical/models")\
    .setInputCols(["document2","token", "word_embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["document2", "token", "ner"])\
    .setOutputCol("ner_chunk")

de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "document2"]) \
    .setOutputCol("deid_text") \
    .setMode("obfuscate") \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setLanguage("en") \
    .setObfuscateRefSource('faker') \
    .setUseShifDays(True)

pipeline_col = Pipeline().setStages([
    documentAssembler,
    documentHasher,
    tokenizer,
    embeddings,
    clinical_ner,
    ner_converter,
    de_identification
    
])

empty_data = spark.createDataFrame([["", "", ""]]).toDF("patientID","text", "dateshift")

pipeline_col_model = pipeline_col.fit(empty_data)

# COMMAND ----------

output = pipeline_col_model.transform(my_input_df)

output.select('text', 'dateshift', 'deid_text.result').show(truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, dates were shifted according to `dateshift` column

# COMMAND ----------

# MAGIC %md
# MAGIC # Not to Deidentify a Part of an Entity
# MAGIC 
# MAGIC Sometimes we may want not to deidentify some entities. For example, according to the HIPAA rules, we don't have to deidentify years. So lets show an example how we can skip the deidentification of an entity. 
# MAGIC 
# MAGIC Pretrained NER models deidentify years as `DATE`. So we will create a contextual parser for extracting `YEAR` entities only, and will merge its results with NER results by using `setBlackList` parameter.
# MAGIC 
# MAGIC You can check [ContextualParser Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.2.Contextual_Parser_Rule_Based_NER.ipynb) to understand its logic better.

# COMMAND ----------

year = {
  "entity": "YEAR",
  "ruleScope": "sentence",
  "matchScope":"token",
  "regex": "^[12][0-9]{3}$",
  "valuesDefinition":[],
#   "prefix": ["in"],
  "contextLength": 100,
  "context": []
}

with open('/dbfs/databricks/driver/year.json', 'w') as f:
    json.dump(year, f)

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

year_contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("entity_year") \
    .setJsonPath("/dbfs/databricks/driver/year.json") \
    .setCaseSensitive(False) \
    .setPrefixAndSuffixMatch(False)

year_model = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, year_contextual_parser]).fit(spark.createDataFrame([[""]]).toDF("text"))

sample_text = "Patient ID: 6515426. My brother was admitted to the hospital in 2005. I will go to spain in 2025. "

lyear = LightPipeline(year_model)
lyear.annotate(sample_text)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, 2005 and 2025 year entities are extracted as `YEAR`. 
# MAGIC 
# MAGIC You can define stronger contextual parser for detecting year entities by  setting a stronger regex rule or playing with the parameters of CP (like `prefix -> in`)

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_generic")

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner_generic"])\
    .setOutputCol("ner_generic_chunk")

year_contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("entity_year") \
    .setJsonPath("/dbfs/databricks/driver/year.json") \
    .setCaseSensitive(False) \
    .setPrefixAndSuffixMatch(False)

chunks_year= ChunkConverter()\
    .setInputCols("entity_year")\
    .setOutputCol("chunk_year")

# First we will merge chunk_year and ner_generic_chunk
# chunk merger will give the precedence to year_chunk
# if they extract the same year entities. 
chunk_merge_1 = ChunkMergeApproach()\
    .setInputCols("chunk_year","ner_generic_chunk")\
    .setOutputCol("deid_merged_chunk")


deid_ner_enriched = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_subentity")

# we will block DATE entities coming from this model
# the dates will already been detected by ner_deid_generic_augmented
ner_converter_enriched = NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner_subentity"]) \
    .setOutputCol("ner_subentity_chunk")\
    .setBlackList(['DATE'])

# now we will merge ner_subentity_chunk with deid_merged_chunk
# and will block YEAR entities 
# in this way YEAR entities will not appear in the results
chunk_merge_2 = ChunkMergeApproach()\
    .setInputCols("deid_merged_chunk","ner_subentity_chunk")\
    .setOutputCol("final_merged_chunk")\
    .setBlackList(["YEAR"])

deid_entity_labels= DeIdentification()\
    .setInputCols(["sentence", "token", "final_merged_chunk"])\
    .setOutputCol("deid_entity_label")\
    .setMode("mask")\
    .setReturnEntityMappings(True)\
    .setMaskingPolicy("entity_labels")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter,
      year_contextual_parser,
      chunks_year,
      chunk_merge_1,
      deid_ner_enriched,
      ner_converter_enriched,
      chunk_merge_2,
      deid_entity_labels
      ])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the sample text below, and we wait single YEAR entities (2008 and 2009) not to deidentify. But if there is months or days with year, they will be deidentified in any case.

# COMMAND ----------

sample_text = """A 28 year old female with a history of gestational diabetes mellitus diagnosed eight years prior to 
presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis 
three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index 
( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting.
Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . 
She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . 
She had been on dapagliflozin since May 2006 . On 30 June 2007 , her physical examination on presentation was 
significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , 
or rigidity . In 2008 laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , 
anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin 
( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed 
as blood samples kept hemolyzing due to significant lipemia .
The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior 
to admission in 2009. However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , 
the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 950 mg/dL , 
and lipase was 52 U/L . She was discharged on 05 June 2012 . 

Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . 
At birth the typical boy is growing slightly faster than the typical girl, but the velocities become equal at about 
seven months, and then the girl grows faster until four years. 
From then until adolescence no differences in velocity 
can be detected. 21-02-2020 
21/04/2020
"""

# COMMAND ----------

lmodel= LightPipeline(model)
lresult= lmodel.fullAnnotate(sample_text)[0]

# COMMAND ----------

# YEAR chunks detected by CP

lresult["chunk_year"]

# COMMAND ----------

# chunks are detected by ner_deid_generic_augmented

lresult["ner_generic_chunk"]

# COMMAND ----------

# 2008 and 2009 entities are labelled as YEAR in merged chunk

lresult["deid_merged_chunk"]

# COMMAND ----------

lresult["final_merged_chunk"]

# COMMAND ----------

# MAGIC %md
# MAGIC `YEAR` entities were in `deid_merged_chunk`, but they are not in `final_merged_chunk`.

# COMMAND ----------

result_df = model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))

# COMMAND ----------

pd.set_option("display.max_colwidth",0)

pd_result = result_df.select(F.explode(F.arrays_zip(result_df.sentence.result, result_df.deid_entity_label.result)).alias("cols")) \
                     .select(F.expr("cols['0']").alias("sentence"),
                             F.expr("cols['1']").alias("deid_entity_label")).toPandas()

pd_result

# COMMAND ----------

# MAGIC %md
# MAGIC When you check the 6th and 9th lines, you can see that the YEAR only entities are not deidentified.

# COMMAND ----------

# MAGIC %md
# MAGIC # Structured Deidentification

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/hipaa-table-001.txt
  
dbutils.fs.cp("file:/databricks/driver/hipaa-table-001.txt", "dbfs:/")

# COMMAND ----------

df = spark.read.format("csv") \
    .option("sep", "\t") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .load("/hipaa-table-001.txt")

df = df.withColumnRenamed("PATIENT","NAME")
df.show(truncate=False)

# COMMAND ----------

from sparknlp_jsl.structured_deidentification import StructuredDeidentification

# COMMAND ----------

obfuscator = StructuredDeidentification(spark,{"NAME":"PATIENT","AGE":"AGE"}, obfuscateRefSource = "faker")
obfuscator_df = obfuscator.obfuscateColumns(df)
obfuscator_df.select("NAME","AGE").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Setting obfuscateRefSource parameter as "file"

# COMMAND ----------

obfuscator_unique_ref_test = '''Will Perry#PATIENT
John Smith#PATIENT
Marvin MARSHALL#PATIENT
Hubert GROGAN#PATIENT
ALTHEA COLBURN#PATIENT
Kalil AMIN#PATIENT
Inci FOUNTAIN#PATIENT
Aberdeen#CITY
Louisburg St#STREET
France#LOC
Nick Riviera#DOCTOR
5552312#PHONE
St James Hospital#HOSPITAL
Calle del Libertador#ADDRESS
111#ID
Will#DOCTOR
20#AGE
30#AGE
40#AGE
50#AGE
60#AGE
'''

with open('/dbfs/obfuscator_unique_ref_test.txt', 'w') as f:
  f.write(obfuscator_unique_ref_test)

# COMMAND ----------

# obfuscateRefSource = "file"

obfuscator = StructuredDeidentification(spark,{"NAME":"PATIENT","AGE":"AGE"}, 
                                        obfuscateRefFile="/dbfs/obfuscator_unique_ref_test.txt", 
                                        obfuscateRefSource = "file",
                                        columnsSeed={"NAME": 23, "AGE": 23})

obfuscator_df = obfuscator.obfuscateColumns(df)
obfuscator_df.select("NAME","AGE").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC We can shift n days in the structured deidentification through "days" parameter when the column is a Date.

# COMMAND ----------

df = spark.createDataFrame([
            ["Juan García", "13/02/1977", "711 Nulla St.", "140", "673 431234"],
            ["Will Smith", "23/02/1977", "1 Green Avenue.", "140", "+23 (673) 431234"],
            ["Pedro Ximénez", "11/04/1900", "Calle del Libertador, 7", "100", "912 345623"]
        ]).toDF("NAME", "DOB", "ADDRESS", "SBP", "TEL")
df.show(truncate=False)

# COMMAND ----------

obfuscator = StructuredDeidentification(spark=spark, columns={"NAME": "ID", "DOB": "DATE"},
                                                     columnsSeed={"NAME": 23, "DOB": 23},
                                                     obfuscateRefSource="faker",
                                                     days=5
                                         )

# COMMAND ----------

result = obfuscator.obfuscateColumns(df)
result.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pretrained Deidentification Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC `clinical_deidentification` pipeline can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`, `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR` entities. There is also a slim version of deidentificaiton pipeline, `clinical_deidentification_slim`

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")

# COMMAND ----------

deid_pipeline.model.stages

# COMMAND ----------

text= """Name : Hendrickson, Ora, Record date: 2093-01-13, Age: 25, # 719435. Dr. John Green, ID: 1231511863, IP 203.120.223.13. He is a 60-year-old male was admitted to the Day Hospital for cystectomy on 01/13/93. Patient's VIN : 1HGBH41JXMN109286, SSN #333-44-6666, Driver's license no:A334455B. Phone (302) 786-5227, 0295 Keats Street, San Francisco."""

# COMMAND ----------

deid_res = deid_pipeline.annotate(text)

# COMMAND ----------

deid_res.keys()

# COMMAND ----------

pd.set_option("display.max_colwidth", 100)

df= pd.DataFrame(list(zip(deid_res["sentence"], 
                          deid_res["masked"],
                          deid_res["masked_with_chars"],
                          deid_res["masked_fixed_length_chars"], 
                          deid_res["obfuscated"])),
                 columns= ["Sentence", "Masked", "Masked with Chars", "Masked with Fixed Chars", "Obfuscated"])

df