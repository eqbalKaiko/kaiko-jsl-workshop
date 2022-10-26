# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC #  Zero-Shot Named Entity Recognition in Spark NLP
# MAGIC 
# MAGIC In this notebook, you will find an example of Zero-Shot NER model (`zero_shot_ner_roberta`) that is the first of its kind and can detect any named entities without using any annotated dataset to train a model. 
# MAGIC 
# MAGIC `ZeroShotNerModel` annotator also allows extracting entities by crafting appropriate prompts to query **any RoBERTa Question Answering model**. 
# MAGIC 
# MAGIC 
# MAGIC You can check the model card here: [Models Hub](https://nlp.johnsnowlabs.com/2022/08/29/zero_shot_ner_roberta_en.html)

# COMMAND ----------

import sparknlp
import sparknlp_jsl
import pandas as pd
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from pyspark.sql import functions as F
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql.types import StringType

print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Zero-Shot Clinical NER Pipeline
# MAGIC 
# MAGIC Now we will create a pipeline for Zero-Shot NER model with only `documentAssembler`, `sentenceDetector`, `tokenizer`, `zero_shot_ner` and `ner_converter` stages. As you can see, we don't use any embeddings model, because it is already included in the model. 
# MAGIC 
# MAGIC Only the thing that you need to do is create meaningful definitions for the entities that you want to extract. For example; we want to detect `PROBLEM`, `DRUG`, `PATIENT_AGE` and  `ADMISSION_DATE` entities, so we created a dictionary with the questions for detecting these entities and the labels that we want to see in the result. Then we provided this dictionary to the model by using `setEntityDefinitions` parameter.

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
    
zero_shot_ner = ZeroShotNerModel.pretrained("zero_shot_ner_roberta", "en", "clinical/models")\
    .setEntityDefinitions(
        {
            "PROBLEM": ["What is the disease?", "What is his symptom?", "What is her disease?", "What is his disease?", 
                        "What is the problem?" ,"What does a patient suffer", 'What was the reason that the patient is admitted to the clinic?'],
            "DRUG": ["Which drug?", "Which is the drug?", "What is the drug?", "Which drug does he use?", "Which drug does she use?", "Which drug do I use?", "Which drug is prescribed for a symptom?"],
            "ADMISSION_DATE": ["When did patient admitted to a clinic?"],
            "PATIENT_AGE": ["How old is the patient?",'What is the gae of the patient?']
        })\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setPredictionThreshold(0.1) # default 0.01

ner_converter = sparknlp.annotators.NerConverter()\
    .setInputCols(["sentence", "token", "zero_shot_ner"])\
    .setOutputCol("ner_chunk")\

pipeline = Pipeline(stages = [
    documentAssembler, 
    sentenceDetector, 
    tokenizer, 
    zero_shot_ner, 
    ner_converter])

zero_shot_ner_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

# COMMAND ----------

zero_shot_ner.extractParamMap()

# COMMAND ----------

zero_shot_ner.getPredictionThreshold()

# COMMAND ----------

text_list = ["The doctor pescribed Majezik for my severe headache.",
             "The patient was admitted to the hospital for his colon cancer.",
             "27 years old patient was admitted to clinic on Sep 1st by Dr. X for a right-sided pleural effusion for thoracentesis."
            ]

data = spark.createDataFrame(text_list, StringType()).toDF("text")

results = zero_shot_ner_model.transform(data)

# COMMAND ----------

results.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Lets check the NER model results.

# COMMAND ----------

results.selectExpr("explode(zero_shot_ner) AS entity")\
       .select(
           "entity.metadata.word",    
           "entity.result",    
           "entity.metadata.sentence",
           "entity.begin",
           "entity.end",
           "entity.metadata.confidence",
           "entity.metadata.question")\
       .show(100, truncate=False)

# COMMAND ----------

results.select(F.explode(F.arrays_zip(results.token.result,
                                      results.zero_shot_ner.result, 
                                      results.zero_shot_ner.metadata,
                                      results.zero_shot_ner.begin, 
                                      results.zero_shot_ner.end)).alias("cols"))\
       .select(F.expr("cols['0']").alias("token"),
               F.expr("cols['1']").alias("ner_label"),
               F.expr("cols['2']['sentence']").alias("sentence"),
               F.expr("cols['3']").alias("begin"),
               F.expr("cols['4']").alias("end"),
               F.expr("cols['2']['confidence']").alias("confidence")).show(50, truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will check the NER chunks.

# COMMAND ----------

results.selectExpr("explode(ner_chunk)").show(100, truncate=False)

# COMMAND ----------

results.select(F.explode(F.arrays_zip(results.ner_chunk.result,
                                      results.ner_chunk.metadata)).alias("cols"))\
       .select(F.expr("cols['0']").alias("chunk"),
               F.expr("cols['1']['entity']").alias("ner_label"),
               F.expr("cols['1']['confidence']").alias("confidence")).show(50, truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LightPipelines

# COMMAND ----------

# fullAnnotate in LightPipeline
print (text_list[-1], "\n")

light_model = LightPipeline(zero_shot_ner_model)
light_result = light_model.fullAnnotate(text_list[-1])

chunks = []
entities = []
sentence= []
begin = []
end = []

for n in light_result[0]['ner_chunk']:
        
    begin.append(n.begin)
    end.append(n.end)
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    sentence.append(n.metadata['sentence'])
    
    

df_clinical = pd.DataFrame({'chunks':chunks, 'begin': begin, 'end':end, 
                   'sentence_id':sentence, 'entities':entities})

df_clinical.head(20)

# COMMAND ----------

light_result[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### NER Visualizer
# MAGIC 
# MAGIC For saving the visualization result as html, provide `save_path` parameter in the display function.

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

for i in text_list:

    light_result = light_model.fullAnnotate(i)
    ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', return_html=True)

    # Change color of an entity label
    # visualiser.set_label_colors({'PROBLEM':'#008080', 'DRUG':'#800080', 'PATIENT_AGE':'#808080'})
    # ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', return_html=True)


    # Set label filter
    # ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document',labels=['PROBLEM'], return_html=True)
    
    displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the Model and Load from Disc
# MAGIC 
# MAGIC Now we will save the Zero-Shot NER model and then we will be able to use this model without definitions. So our model will have the same labels that we defined before.

# COMMAND ----------

# save model

zero_shot_ner.write().overwrite().save("dbfs:/databricks/driver/zero_shot_ner_model")

# COMMAND ----------

# load from disc and create a new pipeline

zero_shot_ner_local = ZeroShotNerModel.load("dbfs:/databricks/driver/zero_shot_ner_model")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("zero_shot_ner")

ner_converter_local = sparknlp.annotators.NerConverter()\
    .setInputCols(["sentence", "token", "zero_shot_ner"])\
    .setOutputCol("ner_chunk")\

pipeline_local = Pipeline(stages = [
    documentAssembler, 
    sentenceDetector, 
    tokenizer, 
    zero_shot_ner_local, 
    ner_converter_local])

zero_shot_ner_model_local = pipeline_local.fit(spark.createDataFrame([[""]]).toDF("text"))

# COMMAND ----------

# check the results

local_results = zero_shot_ner_model_local.transform(data)

local_results.select(F.explode(F.arrays_zip(local_results.ner_chunk.result,
                                            local_results.ner_chunk.metadata)).alias("cols"))\
             .select(F.expr("cols['0']").alias("chunk"),
                     F.expr("cols['1']['entity']").alias("ner_label"),
                     F.expr("cols['1']['confidence']").alias("confidence")).show(50, truncate=100)