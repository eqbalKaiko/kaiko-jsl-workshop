# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Adverse Drug Events Detection using Named Entity Recognition, Classification and Assertion Status Models

# COMMAND ----------

# MAGIC %md
# MAGIC `ADE NER`: Extracts ADE and DRUG entities from clinical texts.
# MAGIC 
# MAGIC `ADE Classifier`: CLassify if a sentence is ADE-related (`True`) or not (`False`)
# MAGIC 
# MAGIC We use several datasets to train these models:
# MAGIC 
# MAGIC - Twitter dataset, which is used in paper "`Deep learning for pharmacovigilance: recurrent neural network architectures for labeling adverse drug reactions in Twitter posts`" (https://pubmed.ncbi.nlm.nih.gov/28339747/)
# MAGIC - ADE-Corpus-V2, which is used in paper "`An Attentive Sequence Model for Adverse Drug Event Extraction from Biomedical Text`" (https://arxiv.org/abs/1801.00625) and availe online: https://sites.google.com/site/adecorpus/home/document.
# MAGIC - CADEC dataset, which is sued in paper `Cadec: A corpus of adverse drug event annotations` (https://pubmed.ncbi.nlm.nih.gov/25817970)

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
#from sparknlp.pretrained import ResourceDownloader

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
# MAGIC ## ADE Classifier Pipeline (with a pretrained model)
# MAGIC 
# MAGIC `True` : The sentence is talking about a possible ADE
# MAGIC 
# MAGIC `False` : The sentences doesn't have any information about an ADE.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ADE Classifier with BioBert

# COMMAND ----------

# Annotator that transforms a text column from dataframe into an Annotation ready for NLP
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("sentence")

# Tokenizer splits words in a relevant format for NLP
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

bert_embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

embeddingsSentence = SentenceEmbeddings() \
    .setInputCols(["sentence", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")\
    .setStorageRef('biobert_pubmed_base_cased')

classsifierdl = ClassifierDLModel.pretrained("classifierdl_ade_biobert", "en", "clinical/models")\
    .setInputCols(["sentence", "sentence_embeddings"]) \
    .setOutputCol("class")

ade_clf_pipeline = Pipeline(
    stages=[documentAssembler, 
            tokenizer,
            bert_embeddings,
            embeddingsSentence,
            classsifierdl])


empty_data = spark.createDataFrame([[""]]).toDF("text")
ade_clf_model = ade_clf_pipeline.fit(empty_data)

ade_lp_pipeline = LightPipeline(ade_clf_model)

# COMMAND ----------

text = "I feel a bit drowsy & have a little blurred vision after taking an insulin"

ade_lp_pipeline.annotate(text)['class'][0]

# COMMAND ----------

text="I just took an Advil and have no gastric problems so far."

ade_lp_pipeline.annotate(text)['class'][0]

# COMMAND ----------

# MAGIC %md
# MAGIC As you see `gastric problems` is not detected as `ADE` as it is in a negative context. So, classifier did a good job detecting that.

# COMMAND ----------

text="I just took a Metformin and started to feel dizzy."

ade_lp_pipeline.annotate(text)['class'][0]

# COMMAND ----------

t='''
Always tired, and possible blood clots. I was on Voltaren for about 4 years and all of the sudden had a minor stroke and had blood clots that traveled to my eye. I had every test in the book done at the hospital, and they couldnt find anything. I was completley healthy! I am thinking it was from the voltaren. I have been off of the drug for 8 months now, and have never felt better. I started eating healthy and working out and that has help alot. I can now sleep all thru the night. I wont take this again. If I have the back pain, I will pop a tylonol instead.
'''

ade_lp_pipeline.annotate(t)['class'][0]


# COMMAND ----------

texts = ["I feel a bit drowsy & have a little blurred vision, after taking a pill.",
"I've been on Arthrotec 50 for over 10 years on and off, only taking it when I needed it.",
"Due to my arthritis getting progressively worse, to the point where I am in tears with the agony, gp's started me on 75 twice a day and I have to take it every day for the next month to see how I get on, here goes.",
"So far its been very good, pains almost gone, but I feel a bit weird, didn't have that when on 50."]

for text in texts:

  result = ade_lp_pipeline.annotate(text)

  print (result['class'][0])


# COMMAND ----------

# MAGIC %md
# MAGIC ### ADE Classifier trained with conversational (short) sentences

# COMMAND ----------

# MAGIC %md
# MAGIC This model is trained on short, conversational sentences related to ADE and is supposed to do better on the text that is short and used in a daily context.

# COMMAND ----------

conv_classsifierdl = ClassifierDLModel.pretrained("classifierdl_ade_conversational_biobert", "en", "clinical/models")\
            .setInputCols(["sentence", "sentence_embeddings"]) \
            .setOutputCol("class")

conv_ade_clf_pipeline = Pipeline(
    stages=[documentAssembler, 
            tokenizer,
            bert_embeddings,
            embeddingsSentence,
            conv_classsifierdl])

empty_data = spark.createDataFrame([[""]]).toDF("text")
conv_ade_clf_model = conv_ade_clf_pipeline.fit(empty_data)

conv_ade_lp_pipeline = LightPipeline(conv_ade_clf_model)

# COMMAND ----------

text = "after taking a pill, he denies any pain"

conv_ade_lp_pipeline.annotate(text)['class'][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## ADE NER
# MAGIC 
# MAGIC Extracts `ADE` and `DRUG` entities from text.

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

ade_ner = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ade_ner,
    ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_ner_model = ner_pipeline.fit(empty_data)

ade_ner_lp = LightPipeline(ade_ner_model)

# COMMAND ----------

light_result = ade_ner_lp.fullAnnotate("I feel a bit drowsy & have a little blurred vision, so far no gastric problems. I have been on Arthrotec 50 for over 10 years on and off, only taking it when I needed it. Due to my arthritis getting progressively worse, to the point where I am in tears with the agony, gp's started me on 75 twice a day and I have to take it every day for the next month to see how I get on, here goes. So far its been very good, pains almost gone, but I feel a bit weird, didn't have that when on 50.")

chunks = []
entities = []
begin =[]
end = []

for n in light_result[0]['ner_chunk']:

    begin.append(n.begin)
    end.append(n.end)
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 

import pandas as pd

df = pd.DataFrame({'chunks':chunks, 'entities':entities,
                    'begin': begin, 'end': end})

df

# COMMAND ----------

# MAGIC %md
# MAGIC As you see `gastric problems` is not detected as `ADE` as it is in a negative context. So, NER did a good job ignoring that.

# COMMAND ----------

# MAGIC %md
# MAGIC #### ADE NER with Bert embeddings

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

bert_embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
  
ade_ner_bert = MedicalNerModel.pretrained("ner_ade_biobert", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")

ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    bert_embeddings,
    ade_ner_bert,
    ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_ner_model_bert = ner_pipeline.fit(empty_data)

ade_ner_lp_bert = LightPipeline(ade_ner_model_bert)

# COMMAND ----------

light_result = ade_ner_lp_bert.fullAnnotate("I feel a bit drowsy & have a little blurred vision, so far no gastric problems. I have been on Arthrotec 50 for over 10 years on and off, only taking it when I needed it. Due to my arthritis getting progressively worse, to the point where I am in tears with the agony, gp's started me on 75 twice a day and I have to take it every day for the next month to see how I get on, here goes. So far its been very good, pains almost gone, but I feel a bit weird, didn't have that when on 50.")

chunks = []
entities = []
begin =[]
end = []

for n in light_result[0]['ner_chunk']:

    begin.append(n.begin)
    end.append(n.end)
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 

import pandas as pd

df = pd.DataFrame({'chunks':chunks, 'entities':entities,
                    'begin': begin, 'end': end})

df

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like Bert version of NER returns more entities than clinical embeddings version.

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER and Classifier combined with AssertionDL Model

# COMMAND ----------

assertion_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ass_ner_chunk")\
    .setWhiteList(['ADE'])

biobert_assertion = AssertionDLModel.pretrained("assertion_dl_biobert", "en", "clinical/models") \
    .setInputCols(["sentence", "ass_ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

assertion_ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    bert_embeddings,
    ade_ner_bert,
    ner_converter,
    assertion_ner_converter,
    biobert_assertion])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_ass_ner_model_bert = assertion_ner_pipeline.fit(empty_data)

ade_ass_ner_model_lp_bert = LightPipeline(ade_ass_ner_model_bert)

# COMMAND ----------

import pandas as pd
text = "I feel a bit drowsy & have a little blurred vision, so far no gastric problems. I have been on Arthrotec 50 for over 10 years on and off, only taking it when I needed it. Due to my arthritis getting progressively worse, to the point where I am in tears with the agony, gp's started me on 75 twice a day and I have to take it every day for the next month to see how I get on, here goes. So far its been very good, pains almost gone, but I feel a bit weird, didn't have that when on 50."

print (text)

light_result = ade_ass_ner_model_lp_bert.fullAnnotate(text)[0]

chunks=[]
entities=[]
status=[]

for n,m in zip(light_result['ass_ner_chunk'],light_result['assertion']):
    
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)
        
df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

df

# COMMAND ----------

# MAGIC %md
# MAGIC Looks great ! `gastric problems` is detected as `ADE` and `absent`

# COMMAND ----------

# MAGIC %md
# MAGIC ## ADE models applied to Spark Dataframes

# COMMAND ----------

import pyspark.sql.functions as F

! wget -q	https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/sample_ADE_dataset.csv

dbutils.fs.cp("file:/databricks/driver/sample_ADE_dataset.csv", "dbfs:/")

# COMMAND ----------

ade_DF = spark.read\
                .option("header", "true")\
                .csv("/sample_ADE_dataset.csv")\
                .filter(F.col("label").isin(['True','False']))

ade_DF.show(truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC **With BioBert version of NER** (will be slower but more accurate)

# COMMAND ----------

import pyspark.sql.functions as F

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(['ADE'])

ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    bert_embeddings,
    ade_ner_bert,
    ner_converter])


empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_ner_model = ner_pipeline.fit(empty_data)

result = ade_ner_model.transform(ade_DF)

sample_df = result.select('text','ner_chunk.result')\
                  .toDF('text','ADE_phrases')\
                  .filter(F.size('ADE_phrases')>0).toPandas()

# COMMAND ----------

import pandas as pd
pd.set_option('display.max_colwidth', 0)

# COMMAND ----------

sample_df.sample(20)

# COMMAND ----------

# MAGIC %md
# MAGIC **Doing the same with clinical embeddings version** (faster results)

# COMMAND ----------

import pyspark.sql.functions as F

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")\
  .setWhiteList(['ADE'])

ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ade_ner,
    ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_ner_model = ner_pipeline.fit(empty_data)

result = ade_ner_model.transform(ade_DF)

result.select('text','ner_chunk.result')\
.toDF('text','ADE_phrases').filter(F.size('ADE_phrases')>0)\
.show(truncate=70)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating sentence dataframe (one sentence per row) and getting ADE entities and categories

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")\
      .setExplodeSentences(True)

tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

bert_embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

embeddingsSentence = SentenceEmbeddings() \
      .setInputCols(["sentence", "embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")\
      .setStorageRef('biobert_pubmed_base_cased')

classsifierdl = ClassifierDLModel.pretrained("classifierdl_ade_biobert", "en", "clinical/models")\
      .setInputCols(["sentence", "sentence_embeddings"]) \
      .setOutputCol("class")\
      .setStorageRef('biobert_pubmed_base_cased')

ade_ner = MedicalNerModel.pretrained("ner_ade_biobert", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
  
ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(['ADE'])

ner_clf_pipeline = Pipeline(
    stages=[documentAssembler, 
            sentenceDetector,
            tokenizer,
            bert_embeddings,
            embeddingsSentence,
            classsifierdl,
            ade_ner,
            ner_converter])

ade_Sentences = ner_clf_pipeline.fit(ade_DF)

# COMMAND ----------

import pyspark.sql.functions as F

ade_Sentences.transform(ade_DF).select('sentence.result','ner_chunk.result','class.result')\
                               .toDF('sentence','ADE_phrases','is_ADE').show(truncate=60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a pretrained pipeline with ADE NER, Assertion and Classifer

# COMMAND ----------

# Annotator that transforms a text column from dataframe into an Annotation ready for NLP
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("sentence")

# Tokenizer splits words in a relevant format for NLP
tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

bert_embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

ade_ner = MedicalNerModel.pretrained("ner_ade_biobert", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")\
      .setStorageRef('biobert_pubmed_base_cased')

ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")

assertion_ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ass_ner_chunk")\
      .setWhiteList(['ADE'])

biobert_assertion = AssertionDLModel.pretrained("assertion_dl_biobert", "en", "clinical/models") \
      .setInputCols(["sentence", "ass_ner_chunk", "embeddings"]) \
      .setOutputCol("assertion")

embeddingsSentence = SentenceEmbeddings() \
      .setInputCols(["sentence", "embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")\
      .setStorageRef('biobert_pubmed_base_cased')

classsifierdl = ClassifierDLModel.pretrained("classifierdl_ade_conversational_biobert", "en", "clinical/models")\
      .setInputCols(["sentence", "sentence_embeddings"]) \
      .setOutputCol("class")

ade_clf_pipeline = Pipeline(
    stages=[documentAssembler, 
            tokenizer,
            bert_embeddings,
            ade_ner,
            ner_converter,
            assertion_ner_converter,
            biobert_assertion,
            embeddingsSentence,
            classsifierdl])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_ner_clf_model = ade_clf_pipeline.fit(empty_data)

ade_ner_clf_pipeline = LightPipeline(ade_ner_clf_model)

# COMMAND ----------

classsifierdl.getStorageRef()

# COMMAND ----------

text = 'Always tired, and possible blood clots. I was on Voltaren for about 4 years and all of the sudden had a minor stroke and had blood clots that traveled to my eye. I had every test in the book done at the hospital, and they couldnt find anything. I was completley healthy! I am thinking it was from the voltaren. I have been off of the drug for 8 months now, and have never felt better. I started eating healthy and working out and that has help alot. I can now sleep all thru the night. I wont take this again. If I have the back pain, I will pop a tylonol instead.'

light_result = ade_ner_clf_pipeline.fullAnnotate(text)

print (light_result[0]['class'][0].metadata)

chunks = []
entities = []
begin =[]
end = []

for n in light_result[0]['ner_chunk']:

    begin.append(n.begin)
    end.append(n.end)
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 

import pandas as pd

df = pd.DataFrame({'chunks':chunks, 'entities':entities,
                    'begin': begin, 'end': end})

df

# COMMAND ----------

import pandas as pd

text = 'I have always felt tired, but no blood clots. I was on Voltaren for about 4 years and all of the sudden had a minor stroke and had blood clots that traveled to my eye. I had every test in the book done at the hospital, and they couldnt find anything. I was completley healthy! I am thinking it was from the voltaren. I have been off of the drug for 8 months now, and have never felt better. I started eating healthy and working out and that has help alot. I can now sleep all thru the night. I wont take this again. If I have the back pain, I will pop a tylonol instead.'

print (text)

light_result = ade_ass_ner_model_lp_bert.fullAnnotate(text)[0]

chunks=[]
entities=[]
status=[]

for n,m in zip(light_result['ass_ner_chunk'],light_result['assertion']):
    
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)
        
df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

df

# COMMAND ----------

result = ade_ner_clf_pipeline.annotate('I just took an Advil 100 mg and it made me drowsy')

print (result['class'])
print(list(zip(result['token'],result['ner'])))

# COMMAND ----------

ade_ner_clf_model.write().overwrite().save('/databricks/driver/ade_pretrained_pipeline')

# COMMAND ----------

ade_ner_clf_model.stages

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

ade_pipeline = PretrainedPipeline.from_disk('/databricks/driver/ade_pretrained_pipeline')

ade_pipeline.annotate('I just took an Advil 100 mg and it made me drowsy')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pretrained ADE Pipeline
# MAGIC 
# MAGIC A pipeline for `Adverse Drug Events (ADE)` with `ner_ade_healthcare`, and `classifierdl_ade_biobert`. It will extract `ADE` and `DRUG` clinical entities, and then assign ADE status to a text(`True` means ADE, `False` means not related to ADE). Also extracts relations between `DRUG` and `ADE` entities (`1` means the adverse event and drug entities are related, `0` is not related).

# COMMAND ----------

pretrained_ade_pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

# COMMAND ----------

pretrained_ade_pipeline.model.stages

# COMMAND ----------

result = pretrained_ade_pipeline.fullAnnotate("The main adverse effects of Leflunomide consist of diarrhea, nausea, liver enzyme elevation, hypertension, alopecia, and allergic skin reactions.")

result[0].keys()

# COMMAND ----------

result[0]['class'][0].metadata

# COMMAND ----------

text = "Jaw,neck, low back and hip pains. Numbness in legs and arms. Took about a month for the same symptoms to begin with Vytorin. The pravachol started the pains again in about 3 months. I stopped taking all statin drungs. Still hurting after 2 months of stopping. Be careful taking this drug."

import pandas as pd

chunks = []
entities = []
begin =[]
end = []

print ('sentence:', text)
print()

result = pretrained_ade_pipeline.fullAnnotate(text)

print ('ADE status:', result[0]['class'][0].result)

print ('prediction probability>> True : ', result[0]['class'][0].metadata['True'], \
        'False: ', result[0]['class'][0].metadata['False'])

for n in result[0]['ner_chunks_ade']:

    begin.append(n.begin)
    end.append(n.end)
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 

df = pd.DataFrame({'chunks':chunks, 'entities':entities,
                'begin': begin, 'end': end})

df


# COMMAND ----------

# MAGIC %md
# MAGIC #### with AssertionDL

# COMMAND ----------

import pandas as pd

text = """I have an allergic reaction to vancomycin. 
My skin has be itchy, sore throat/burning/itchy, and numbness in tongue and gums. 
I would not recommend this drug to anyone, especially since I have never had such an adverse reaction to any other medication."""

print (text)

light_result = pretrained_ade_pipeline.fullAnnotate(text)[0]

chunks=[]
entities=[]
status=[]

for n,m in zip(light_result['ner_chunks_ade_assertion'],light_result['assertion_ade']):
    
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)
        
df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

df

# COMMAND ----------

# MAGIC %md
# MAGIC #### with Relation Extraction

# COMMAND ----------

import pandas as pd

text = """I have Rhuematoid Arthritis for 35 yrs and have been on many arthritis meds. 
I currently am on Relefen for inflamation, Prednisone 5mg, every other day and Enbrel injections once a week. 
I have no problems from these drugs. Eight months ago, another doctor put me on Lipitor 10mg daily because my chol was 240. 
Over a period of 6 months, it went down to 159, which was great, BUT I started having terrible aching pain in my arms about that time which was radiating down my arms from my shoulder to my hands.
"""
 
print (text)

results = pretrained_ade_pipeline.fullAnnotate(text)

rel_pairs=[]

for rel in results[0]["relations_ade_drug"]:
    rel_pairs.append((
        rel.result, 
        rel.metadata['entity1'], 
        rel.metadata['entity1_begin'],
        rel.metadata['entity1_end'],
        rel.metadata['chunk1'], 
        rel.metadata['entity2'],
        rel.metadata['entity2_begin'],
        rel.metadata['entity2_end'],
        rel.metadata['chunk2'], 
        rel.metadata['confidence']
    ))

rel_df = pd.DataFrame(rel_pairs, columns=['relation','entity1','entity1_begin','entity1_end','chunk1','entity2','entity2_begin','entity2_end','chunk2', 'confidence'])
rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC You can check the links below if you want to see more examples of; 
# MAGIC 
# MAGIC - Pretrained Clinical Pipelines in Spark NLP : [11.Pretrained Clinical Pipelines](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb)
# MAGIC - Relation Extraction Model of DRUG and ADE entities `re_ade_biobert`: [10. Clinical Relation Extraction Model](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb)

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #