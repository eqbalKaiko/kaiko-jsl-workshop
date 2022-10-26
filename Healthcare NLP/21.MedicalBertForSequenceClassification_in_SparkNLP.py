# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # MedicalBertForSequenceClassification Models

# COMMAND ----------

# MAGIC %md
# MAGIC 🔎 `MedicalBertForTokenClassification` annotator can load BERT Models with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.
# MAGIC 
# MAGIC Transformer models are compatible with Spark NLP library. So if you want to train and import a transformer based `BertForSequenceClassification` model to Spark NLP, you can do it by using [Import Transformers Into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/15.Import_Transformers_Into_Spark_NLP.ipynb) notebook. **BUT**, don't forget to use `BertForSequenceClassification` open source annotator while importing.

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 Pretrained models can be loaded with pretrained() of the companion object.
# MAGIC 
# MAGIC 📌 The default model is `bert_sequence_classifier_ade`, if no name is provided.

# COMMAND ----------

# MAGIC %md
# MAGIC <center><b> MODEL LIST </b>
# MAGIC 
# MAGIC | model name                                                                                                                                                                    | model name | model name |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC |<b>🔎PUBLIC HEALTH</b> |<b>🔎CLINICAL TRIALS</b>|<b>🔎MBFSC Models in Other Languages</b>|
# MAGIC |<b>MBFSC Models related to Mentions and Statements👇🏻</b>|<b> MBFSC Models related to RCT(Randomize Clinical Trials👇🏻</b>|<b>German:👇🏻</b> |
# MAGIC |[bert_sequence_classifier_health_mentions](https://nlp.johnsnowlabs.com/2022/07/25/bert_sequence_classifier_health_mentions_en_3_0.html)| [bert_sequence_classifier_rct_biobert](https://nlp.johnsnowlabs.com/2022/03/01/bert_sequence_classifier_rct_biobert_en_3_0.html)|[bert_sequence_classifier_health_mentions_gbert_large](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_gbert_large_de_3_0.html)|
# MAGIC |[bert_sequence_classifier_question_statement_clinical](https://nlp.johnsnowlabs.com/2021/11/05/bert_sequence_classifier_question_statement_clinical_en.html)|[bert_sequence_classifier_binary_rct_biobert](https://nlp.johnsnowlabs.com/2022/04/25/bert_sequence_classifier_binary_rct_biobert_en_3_0.html)             |[bert_sequence_classifier_health_mentions_bert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_bert_de_3_0.html) |
# MAGIC |<b>MBFSC Models related to Covid👇🏻</b>| |  [bert_sequence_classifier_health_mentions_gbert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_gbert_de_3_0.html)|
# MAGIC |[bert_sequence_classifier_covid_sentiment](https://nlp.johnsnowlabs.com/2022/08/01/bert_sequence_classifier_covid_sentiment_en_3_0.html)| |[bert_sequence_classifier_health_mentions_medbert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_medbert_de_3_0.html)|
# MAGIC |[bert_sequence_classifier_vaccine_sentiment](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_vaccine_sentiment_en_3_0.html)             | |<b>Spanish:👇🏻</b> |
# MAGIC |[bert_sequence_classifier_self_reported_vaccine_status_tweet](https://nlp.johnsnowlabs.com/2022/07/29/bert_sequence_classifier_self_reported_vaccine_status_tweet_en_3_0.html)     |        |[bert_sequence_classifier_self_reported_symptoms_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_self_reported_symptoms_tweet_es_3_0.html)|
# MAGIC |[bert_sequence_classifier_health_mandates_stance_tweet](https://nlp.johnsnowlabs.com/2022/08/08/bert_sequence_classifier_health_mandates_stance_tweet_en_3_0.html)             | 
# MAGIC |[bert_sequence_classifier_health_mandates_premise_tweet](https://nlp.johnsnowlabs.com/2022/08/08/bert_sequence_classifier_health_mandates_premise_tweet_en_3_0.html)             | 
# MAGIC |<b>MBFSC Models related to Mental, Emotional and Social Relationships👇🏻</b>                                                                                                            | 
# MAGIC |[bert_sequence_classifier_stress](https://nlp.johnsnowlabs.com/2022/06/28/bert_sequence_classifier_stress_en_3_0.html)|
# MAGIC |[bert_sequence_classifier_stressor](https://nlp.johnsnowlabs.com/2022/07/27/bert_sequence_classifier_stressor_en_3_0.html)| |
# MAGIC |[bert_sequence_classifier_self_reported_stress_tweet](https://nlp.johnsnowlabs.com/2022/07/29/bert_sequence_classifier_self_reported_stress_tweet_en_3_0.html )|
# MAGIC |[bert_sequence_classifier_depression](https://nlp.johnsnowlabs.com/2022/08/09/bert_sequence_classifier_depression_en_3_0.html)|
# MAGIC |[bert_sequence_classifier_depression_binary](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_depression_binary_en_3_0.html)|
# MAGIC |[bert_sequence_classifier_depression_twitter](https://nlp.johnsnowlabs.com/2022/08/09/bert_sequence_classifier_depression_twitter_en_3_0.html)|
# MAGIC |[bert_sequence_classifier_self_reported_partner_violence_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_self_reported_partner_violence_tweet_en_3_0.html)|
# MAGIC |<b>MBFSC Models related to Treatment👇🏻 </b>                                                                                                                                                                  |
# MAGIC |[bert_sequence_classifier_ade](https://nlp.johnsnowlabs.com/2022/02/08/bert_sequence_classifier_ade_en.html)                |
# MAGIC |[bert_sequence_classifier_ade_augmented](https://nlp.johnsnowlabs.com/2022/07/27/bert_sequence_classifier_ade_augmented_en_3_0.html)|
# MAGIC |[bert_sequence_classifier_drug_reviews_webmd](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_drug_reviews_webmd_en_3_0.html)|
# MAGIC |[bert_sequence_classifier_treatment_changes_sentiment_tweet](https://nlp.johnsnowlabs.com/2022/08/04/bert_sequence_classifier_treatment_changes_sentiment_tweet_en_3_0.html)|
# MAGIC |<b>MBFSC Models related to Gender and Age👇🏻</b>                                                                                                                                                                   |
# MAGIC |[bert_sequence_classifier_gender_biobert](https://nlp.johnsnowlabs.com/2022/02/08/bert_sequence_classifier_gender_biobert_en.html)                |
# MAGIC |[bert_sequence_classifier_exact_age_reddit](https://nlp.johnsnowlabs.com/2022/07/26/bert_sequence_classifier_exact_age_reddit_en_3_0.html)|
# MAGIC |[bert_sequence_classifier_self_reported_age_tweet](https://nlp.johnsnowlabs.com/2022/07/26/bert_sequence_classifier_self_reported_age_tweet_en_3_0.html)|

# COMMAND ----------

# MAGIC %md
# MAGIC # Colab Setup

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
from sparknlp.pretrained import  PretrainedPipeline
from pyspark.sql.types import StringType,IntegerType

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

import warnings
warnings.filterwarnings('ignore')

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())


spark

# COMMAND ----------

# MAGIC %md
# MAGIC # 🔎PUBLIC HEALTH

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔹 MBFSC Models related to Health Mentions and Statement

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_health_mentions](https://nlp.johnsnowlabs.com/2022/07/25/bert_sequence_classifier_health_mentions_en_3_0.html)                | This model is a PHS-BERT based sequence classification model that can classify public health mentions in social media text. | `health_mention` `other_mention` `figurative_mention`                  |
# MAGIC |[bert_sequence_classifier_question_statement_clinical](https://nlp.johnsnowlabs.com/2021/11/05/bert_sequence_classifier_question_statement_clinical_en.html)| This model makes sentence classification by distinguishing between Questions and Statements in the Clinical domain.      | `question` `statement`|

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌Public Health Mention Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_health_mentions`*

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("classes")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["Another uncle of mine had a heart attack and passed away. Will be cremated Saturday I think I ve gone numb again RIP Uncle Mike"]]).toDF("text")

result = model.transform(data)

result.select("text", "classes.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔎with LightPipeline

# COMMAND ----------

# MAGIC %md
# MAGIC 📌Let's show how we can use our same model as LightPipeline

# COMMAND ----------

sample_text = "I don't wanna fall in love. If I ever did that, I think I'd have a heart attack"

light_model = LightPipeline(model)
light_result = light_model.fullAnnotate(sample_text)

# COMMAND ----------

light_result[0]['classes'][0].result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔹 MBFSC Models Related to Covid

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_covid_sentiment](https://nlp.johnsnowlabs.com/2022/08/01/bert_sequence_classifier_covid_sentiment_en_3_0.html)                                           | This model is a BioBERT based sentiment analysis model that can extract information from COVID-19 pandemic-related tweets.                       | `neutral` `positive` `negative`                                              |
# MAGIC |[bert_sequence_classifier_vaccine_sentiment](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_vaccine_sentiment_en_3_0.html)             | This model is a BioBERT based sentimental analysis model that can extract information from COVID-19 Vaccine-related tweets.                            | `neutral` `positive` `negative`                                   |
# MAGIC |[bert_sequence_classifier_self_reported_vaccine_status_tweet](https://nlp.johnsnowlabs.com/2022/07/29/bert_sequence_classifier_self_reported_vaccine_status_tweet_en_3_0.html)             | Classification of tweets indicating self-reported COVID-19 vaccination status. This model involves the identification of self-reported COVID-19 vaccination status in English tweets.                           | `Vaccine_chatter` `Self_reports`                                   |
# MAGIC |[bert_sequence_classifier_health_mandates_stance_tweet](https://nlp.johnsnowlabs.com/2022/08/08/bert_sequence_classifier_health_mandates_stance_tweet_en_3_0.html)             | This model is a BioBERT based classifier that can classify stance about health mandates related to Covid-19 from tweets.  | `Support` `Disapproval` `Not stated `                                   |
# MAGIC |[bert_sequence_classifier_health_mandates_premise_tweet](https://nlp.johnsnowlabs.com/2022/08/08/bert_sequence_classifier_health_mandates_premise_tweet_en_3_0.html)             | This model is a BioBERT based classifier that can classify premise about health mandates related to Covid-19 from tweets. | `Has premise (argument)` `Has no premise (no argument)`                                  |

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌Covid-19 Sentiment Classification Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_covid_sentiment`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_covid_sentiment", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
    ["British Department of Health confirms first two cases of in UK"],
    ["so my trip to visit my australian exchange student just got canceled bc of coronavirus. im heartbroken :("], 
    [ "I wish everyone to be safe at home and stop pandemic"]]
).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔹 MBFSC Models related to Stress, Depression and Partner Violence

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_stress](https://nlp.johnsnowlabs.com/2022/06/28/bert_sequence_classifier_stress_en_3_0.html)| This model is a PHS-BERT-based classifier that can classify whether the content of a text expresses emotional stress.         | `stress` `no-stress`|
# MAGIC |[bert_sequence_classifier_stressor](https://nlp.johnsnowlabs.com/2022/07/27/bert_sequence_classifier_stressor_en_3_0.html)| This model is a bioBERT based classifier that can classify source of emotional stress in text.        | `Family_Issues` `Financial_Problem` `Health_Fatigue_or_Physical` `Pain` `Other``School` `Work` `Social_Relationships`|
# MAGIC |[bert_sequence_classifier_self_reported_stress_tweet](https://nlp.johnsnowlabs.com/2022/07/29/bert_sequence_classifier_self_reported_stress_tweet_en_3_0.html )                | This model classifies stress in social media (Twitter) posts in the self-disclosure category.| `stressed` `not-stressed`                   |
# MAGIC |[bert_sequence_classifier_depression](https://nlp.johnsnowlabs.com/2022/08/09/bert_sequence_classifier_depression_en_3_0.html)| This model is a PHS-BERT based text classification model that can classify depression level of social media text into three levels.         | `no-depression` `minimum` `high-depression`|
# MAGIC |[bert_sequence_classifier_depression_binary](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_depression_binary_en_3_0.html)| This model is a PHS-BERT based text classification model that can classify whether a social media text expresses depression or not.          | `no-depression` `depression`|
# MAGIC |[bert_sequence_classifier_depression_twitter](https://nlp.johnsnowlabs.com/2022/08/09/bert_sequence_classifier_depression_twitter_en_3_0.html)| This model is a PHS-BERT based tweet classification model that can classify whether tweets contain depressive text.        | `no-depression` `depression`|
# MAGIC |[bert_sequence_classifier_self_reported_partner_violence_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_self_reported_partner_violence_tweet_en_3_0.html)| This model classifies self-reported Intimate partner violence (IPV) in tweets.          | `intimate_partner_violence` `non_intimate_partner_violence`|

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌 Depression Level Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_depression`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
             ["None that I know of. Any mental health issue needs to be cared for like any other health issue. Doctors and medications can help."], 
             ["I don’t know. Was this okay? Should I hate him? Or was it just something new? I really don’t know what to make of the situation."], 
             ["It makes me so disappointed in myself because I hate what I've become and I hate feeling so helpless."]
    ]).toDF("text")

result = pipeline.fit(data).transform(data)
result.select("text", "class.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌Self Reported Partner Violence Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_self_reported_partner_violence_tweet`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_partner_violence_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

example = spark.createDataFrame(["I am fed up with this toxic relation.I hate my husband.",
                                 "Can i say something real quick I ve never been one to publicly drag an ex partner and sometimes I regret that. I ve been reflecting on the harm, abuse and violence that was done to me and those bitches are truly lucky I chose peace amp therapy because they are trash forreal."], StringType()).toDF("text")

result = pipeline.fit(example).transform(example)

result.select("text", "class.result").show(truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔹 MBFSC Models related to Treatment

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_ade](https://nlp.johnsnowlabs.com/2022/02/08/bert_sequence_classifier_ade_en.html)                | This model is a BioBERT based classifier that classifies texts/sentences into two categories as True and False.| `True` `False`                   |
# MAGIC |[bert_sequence_classifier_ade_augmented](https://nlp.johnsnowlabs.com/2022/07/27/bert_sequence_classifier_ade_augmented_en_3_0.html)| This model is a [BioBERT-based] (https://github.com/dmis-lab/biobert) classifier that can classify tweets reporting ADEs (Adverse Drug Events).        | `ADE` `no-ADE`|
# MAGIC |[bert_sequence_classifier_drug_reviews_webmd](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_drug_reviews_webmd_en_3_0.html)| This model is a BioBERT based classifier that can classify drug reviews from WebMD.com         | `negative` `positive`|
# MAGIC |[bert_sequence_classifier_treatment_changes_sentiment_tweet](https://nlp.johnsnowlabs.com/2022/08/04/bert_sequence_classifier_treatment_changes_sentiment_tweet_en_3_0.html)| This model is a BioBERT based classifier that can classify patients non-adherent to their treatments and their reasons on Twitter.        | `negative` `positive`|

# COMMAND ----------

# MAGIC %md
# MAGIC > ## 📌Adverse Drug Event Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_ade_augmented`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_ade_augmented", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame(["So glad I am off effexor, so sad it ruined my teeth. tip Please be carefull taking antideppresiva and read about it 1st",
                              "Religare Capital Ranbaxy has been accepting approval for Diovan since 2012"], StringType()).toDF("text")
              
result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌Treatment Changes Sentiment Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_treatment_changes_sentiment_tweet`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_treatment_changes_sentiment_tweet", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame([
                             ["I love when they say things like this. I took that ambien instead of my thyroid pill."],
                             ["I am a 30 year old man who is not overweight but is still on the verge of needing a Lipitor prescription."]
                             ]).toDF("text")
                          
result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔹 MBFSC Models related to Gender and Age

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_gender_biobert](https://nlp.johnsnowlabs.com/2022/02/08/bert_sequence_classifier_gender_biobert_en.html)                | This model classifies the gender of a patient in a clinical document using context.| `Female` `Male` `Unknown`                  |
# MAGIC |[bert_sequence_classifier_exact_age_reddit](https://nlp.johnsnowlabs.com/2022/07/26/bert_sequence_classifier_exact_age_reddit_en_3_0.html)| This model is a BioBERT based classifier that can classify self-report the exact age into social media forum (Reddit) posts.       | `self_report_age` `no_report`|
# MAGIC |[bert_sequence_classifier_self_reported_age_tweet](https://nlp.johnsnowlabs.com/2022/07/26/bert_sequence_classifier_self_reported_age_tweet_en_3_0.html)| This model is a BioBERT based classifier that can classify self-report the exact age into social media data.        | `self_report_age` `no_report`|

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌Gender Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_gender_biobert`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_gender_biobert", "en", "clinical/models")\
  .setInputCols(["document","token"]) \
  .setOutputCol("class") \
  .setCaseSensitive(True) \
  .setMaxSentenceLength(512)


pipeline = Pipeline(stages=[
      document_assembler, 
      tokenizer,
      sequenceClassifier    
      ])


data = spark.createDataFrame([["The patient took Advil and he experienced an adverse reaction."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 🔎CLINICAL TRIALS

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔹 MBFSC Models related to RCT(Randomized Clinical Trials)

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_rct_biobert](https://nlp.johnsnowlabs.com/2022/03/01/bert_sequence_classifier_rct_biobert_en_3_0.html)                                           | This model is a BioBERT-based classifier that can classify the sections within the abstracts of scientific articles regarding randomized clinical trials (RCT).                       | `BACKGROUND` `CONCLUSIONS` `METHODS` `OBJECTIVE` `RESULTS`                                          |
# MAGIC |[bert_sequence_classifier_binary_rct_biobert](https://nlp.johnsnowlabs.com/2022/04/25/bert_sequence_classifier_binary_rct_biobert_en_3_0.html)             | This model is a BioBERT based classifier that can classify if an article is a randomized clinical trial (RCT) or not.                            | `True` `False                               |

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌RCT Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_rct_biobert`*

# COMMAND ----------

sequenceClassifier_loaded = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_rct_biobert", "en", "clinical/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")


pipeline = Pipeline(stages=[
  document_assembler, 
  tokenizer,
  sequenceClassifier_loaded   
  ])


data = spark.createDataFrame([["""Previous attempts to prevent all the unwanted postoperative responses to major surgery with an epidural hydrophilic opioid , morphine , have not succeeded . The authors ' hypothesis was that the lipophilic opioid fentanyl , infused epidurally close to the spinal-cord opioid receptors corresponding to the dermatome of the surgical incision , gives equal pain relief but attenuates postoperative hormonal and metabolic responses more effectively than does systemic fentanyl ."""]]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC # **☞ MBFSC Models in Other Languages**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔎German

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_health_mentions_gbert_large](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_gbert_large_de_3_0.html)                                           | This model is a GBERT-large based sequence classification model that can classify public health mentions in German social media text.                    | `non-health` `health-related`                                           |
# MAGIC |[bert_sequence_classifier_health_mentions_bert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_bert_de_3_0.html)                                           | This model is a bert-base-german based sequence classification model that can classify public health mentions in German social media text.                     | `non-health` `health-related`                                           |
# MAGIC |[bert_sequence_classifier_health_mentions_gbert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_gbert_de_3_0.html)                                           | This model is a GBERT-base based sequence classification model that can classify public health mentions in German social media text.                    | `non-health` `health-related`                                          |
# MAGIC |[bert_sequence_classifier_health_mentions_medbert](https://nlp.johnsnowlabs.com/2022/08/10/bert_sequence_classifier_health_mentions_medbert_de_3_0.html)                                           | This model is a German-MedBERT based sequence classification model that can classify public health mentions in German social media text.                    |  `non-health` `health-related`                                          |

# COMMAND ----------

# MAGIC %md
# MAGIC > ### 📌Public Health Mention Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_health_mentions_gbert_large`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions_bert", "de", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
      ["Durch jahrelanges Rauchen habe ich meine Lunge einfach zu sehr geschädigt - Punkt."],
      ["die Schatzsuche war das Highlight beim Kindergeburtstag, die kids haben noch lange davon gesprochen"]
      ]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔎Spanish

# COMMAND ----------

# MAGIC %md
# MAGIC | model name                                                                                                                                                                    | description                                                                             | predicted entities                                         |
# MAGIC |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------|
# MAGIC |[bert_sequence_classifier_self_reported_symptoms_tweet](https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_self_reported_symptoms_tweet_es_3_0.html)                                           | This model is a BERT based classifier that can classify the origin of symptoms related to Covid-19 from Spanish tweets.                     | `Lit-News_mentions` `Self_reports` ` non-personal_reports`                                        |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📌Self Reported Symptoms Classifier Model
# MAGIC 
# MAGIC *`bert_sequence_classifier_self_reported_symptoms_tweet`*

# COMMAND ----------

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_symptoms_tweet", "es", "clinical/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("class")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
    ])



data = spark.createDataFrame(["Las vacunas 3 y hablamos inminidad vivo  Son bichito vivo dentro de lÃ­quido de la vacuna suelen tener reacciones alÃorgicas si que sepan",
                              "Yo pense que me estaba dando el  coronavirus porque cuando me levante  casi no podia respirar pero que si era que tenia la nariz topada de mocos.",
                              "Tos, dolor de garganta y fiebre, los síntomas más reportados por los porteños con coronavirus"], StringType()).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)