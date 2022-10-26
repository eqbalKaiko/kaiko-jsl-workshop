# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 10.  Transformers for Token Classification & Sequence  Classification

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.sql import functions as F
import pandas as pd

print("Spark NLP version", sparknlp.version())

print("Apache Spark version:", spark.version)

spark

# COMMAND ----------

# MAGIC %md
# MAGIC # Transformers for Token Classification in Spark NLP

# COMMAND ----------

# MAGIC %md
# MAGIC **BertForTokenClassification** can load Bert Models with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.
# MAGIC 
# MAGIC Pretrained models can be loaded with `pretrained()` of the companion object. The default model is `"bert_base_token_classifier_conll03"`, if no name is provided. <br/><br/>
# MAGIC 
# MAGIC 
# MAGIC ### **Here are Bert Based Token Classification models available in Spark NLP**
# MAGIC <br/>
# MAGIC 
# MAGIC | Title                                                                                                                        | Name                                          | Language   |
# MAGIC |:-----------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|:-----------|
# MAGIC | BERT Token Classification - NER CoNLL (bert_base_token_classifier_conll03)                                                   | bert_base_token_classifier_conll03            | en         |
# MAGIC | BERT Token Classification - NER OntoNotes (bert_base_token_classifier_ontonote)                                              | bert_base_token_classifier_ontonote           | en         |
# MAGIC | BERT Token Classification Large - NER CoNLL (bert_large_token_classifier_conll03)                                            | bert_large_token_classifier_conll03           | en         |
# MAGIC | BERT Token Classification Large - NER OntoNotes (bert_large_token_classifier_ontonote)                                       | bert_large_token_classifier_ontonote          | en         |
# MAGIC | BERT Token Classification - ParsBERT for Persian Language Understanding (bert_token_classifier_parsbert_armanner)            | bert_token_classifier_parsbert_armanner       | fa         |
# MAGIC | BERT Token Classification - ParsBERT for Persian Language Understanding (bert_token_classifier_parsbert_ner)                 | bert_token_classifier_parsbert_ner            | fa         |
# MAGIC | BERT Token Classification - ParsBERT for Persian Language Understanding (bert_token_classifier_parsbert_peymaner)            | bert_token_classifier_parsbert_peymaner       | fa         |
# MAGIC | BERT Token Classification - BETO Spanish Language Understanding (bert_token_classifier_spanish_ner)                          | bert_token_classifier_spanish_ner             | es         |
# MAGIC | BERT Token Classification - Swedish Language Understanding (bert_token_classifier_swedish_ner)                               | bert_token_classifier_swedish_ner             | sv         |
# MAGIC | BERT Token Classification - Turkish Language Understanding (bert_token_classifier_turkish_ner)                               | bert_token_classifier_turkish_ner             | tr         |
# MAGIC | DistilBERT Token Classification - NER CoNLL (distilbert_base_token_classifier_conll03)                                       | distilbert_base_token_classifier_conll03      | en         |
# MAGIC | DistilBERT Token Classification - NER OntoNotes (distilbert_base_token_classifier_ontonotes)                                 | distilbert_base_token_classifier_ontonotes    | en         |
# MAGIC | DistilBERT Token Classification - DistilbertNER for Persian Language Understanding (distilbert_token_classifier_persian_ner) | distilbert_token_classifier_persian_ner       | fa         |
# MAGIC | BERT Token Classification -  Few-NERD (bert_base_token_classifier_few_nerd)                                                  | bert_base_token_classifier_few_nerd           | en         |
# MAGIC | DistilBERT Token Classification -  Few-NERD (distilbert_base_token_classifier_few_nerd)                                      | distilbert_base_token_classifier_few_nerd     | en         |
# MAGIC | Named Entity Recognition for Japanese (BertForTokenClassification)                                                           | bert_token_classifier_ner_ud_gsd              | ja         |
# MAGIC | Detect PHI for Deidentification (BertForTokenClassifier)                                                                     | bert_token_classifier_ner_deid                | en         |
# MAGIC | Detect Clinical Entities (BertForTokenClassifier)                                                                            | bert_token_classifier_ner_jsl                 | en         |
# MAGIC | Detect Drug Chemicals (BertForTokenClassifier)                                                                               | bert_token_classifier_ner_drugs               | en         |
# MAGIC | Detect Clinical Entities (Slim version, BertForTokenClassifier)                                                              | bert_token_classifier_ner_jsl_slim            | en         |
# MAGIC | ALBERT Token Classification Base - NER CoNLL (albert_base_token_classifier_conll03)                                          | albert_base_token_classifier_conll03          | en         |
# MAGIC | ALBERT Token Classification Large - NER CoNLL (albert_large_token_classifier_conll03)                                        | albert_large_token_classifier_conll03         | en         |
# MAGIC | ALBERT Token Classification XLarge - NER CoNLL (albert_xlarge_token_classifier_conll03)                                      | albert_xlarge_token_classifier_conll03        | en         |
# MAGIC | DistilRoBERTa Token Classification - NER OntoNotes (distilroberta_base_token_classifier_ontonotes)                           | distilroberta_base_token_classifier_ontonotes | en         |
# MAGIC | RoBERTa Token Classification Base - NER CoNLL (roberta_base_token_classifier_conll03)                                        | roberta_base_token_classifier_conll03         | en         |
# MAGIC | RoBERTa Token Classification Base - NER OntoNotes (roberta_base_token_classifier_ontonotes)                                  | roberta_base_token_classifier_ontonotes       | en         |
# MAGIC | RoBERTa Token Classification Large - NER CoNLL (roberta_large_token_classifier_conll03)                                      | roberta_large_token_classifier_conll03        | en         |
# MAGIC | RoBERTa Token Classification Large - NER OntoNotes (roberta_large_token_classifier_ontonotes)                                | roberta_large_token_classifier_ontonotes      | en         |
# MAGIC | RoBERTa Token Classification For Persian (roberta_token_classifier_zwnj_base_ner)                                            | roberta_token_classifier_zwnj_base_ner        | fa         |
# MAGIC | XLM-RoBERTa Token Classification Base - NER XTREME (xlm_roberta_token_classifier_ner_40_lang)                                | xlm_roberta_token_classifier_ner_40_lang      | xx         |
# MAGIC | XLNet Token Classification Base - NER CoNLL (xlnet_base_token_classifier_conll03)                                            | xlnet_base_token_classifier_conll03           | en         |
# MAGIC | XLNet Token Classification Large - NER CoNLL (xlnet_large_token_classifier_conll03)                                          | xlnet_large_token_classifier_conll03          | en         |
# MAGIC | Detect Adverse Drug Events (BertForTokenClassification)                                                                      | bert_token_classifier_ner_ade                 | en         |
# MAGIC | Detect Anatomical Regions (BertForTokenClassification)                                                                       | bert_token_classifier_ner_anatomy             | en         |
# MAGIC | Detect Bacterial Species (BertForTokenClassification)                                                                        | bert_token_classifier_ner_bacteria            | en         |
# MAGIC | XLM-RoBERTa Token Classification Base - NER CoNLL (xlm_roberta_base_token_classifier_conll03)                                | xlm_roberta_base_token_classifier_conll03     | en         |
# MAGIC | XLM-RoBERTa Token Classification Base - NER OntoNotes (xlm_roberta_base_token_classifier_ontonotes)                          | xlm_roberta_base_token_classifier_ontonotes   | en         |
# MAGIC | Longformer Token Classification Base - NER CoNLL (longformer_base_token_classifier_conll03)                                  | longformer_base_token_classifier_conll03      | en         |
# MAGIC | Longformer Token Classification Base - NER CoNLL (longformer_large_token_classifier_conll03)                                 | longformer_large_token_classifier_conll03     | en         |
# MAGIC | Detect Chemicals in Medical text (BertForTokenClassification)                                                                | bert_token_classifier_ner_chemicals           | en         |
# MAGIC | Detect Chemical Compounds and Genes (BertForTokenClassifier)                                                                 | bert_token_classifier_ner_chemprot            | en         |
# MAGIC | Detect Cancer Genetics (BertForTokenClassification)                                                                          | bert_token_classifier_ner_bionlp              | en         |
# MAGIC | Detect Cellular/Molecular Biology Entities (BertForTokenClassification)                                                      | bert_token_classifier_ner_cellular            | en         |
# MAGIC | Detect concepts in drug development trials (BertForTokenClassification)                                                      | bert_token_classifier_drug_development_trials | en         |
# MAGIC | Detect Cancer Genetics (BertForTokenClassification)                                                                          | bert_token_classifier_ner_bionlp              | en         |
# MAGIC | Detect Adverse Drug Events (BertForTokenClassification)                                                                      | bert_token_classifier_ner_ade                 | en         |
# MAGIC | Detect Anatomical Regions (MedicalBertForTokenClassifier)                                                                    | bert_token_classifier_ner_anatomy             | en         |
# MAGIC | Detect Cellular/Molecular Biology Entities (BertForTokenClassification)                                                      | bert_token_classifier_ner_cellular            | en         |
# MAGIC | Detect Chemicals in Medical text (BertForTokenClassification)                                                                | bert_token_classifier_ner_chemicals           | en         |
# MAGIC | Detect Chemical Compounds and Genes (BertForTokenClassifier)                                                                 | bert_token_classifier_ner_chemprot            | en         |
# MAGIC | Detect PHI for Deidentification (BertForTokenClassifier)                                                                     | bert_token_classifier_ner_deid                | en         |
# MAGIC | Detect Drug Chemicals (BertForTokenClassifier)                                                                               | bert_token_classifier_ner_drugs               | en         |
# MAGIC | Detect Clinical Entities (BertForTokenClassifier)                                                                            | bert_token_classifier_ner_jsl                 | en         |
# MAGIC | Detect Clinical Entities (Slim version, BertForTokenClassifier)                                                              | bert_token_classifier_ner_jsl_slim            | en         |
# MAGIC | Detect Bacterial Species (BertForTokenClassification)                                                                        | bert_token_classifier_ner_bacteria            | en         |

# COMMAND ----------

# MAGIC %md
# MAGIC ## BertForTokenClassification Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's create a Spark NLP Pipeline with `bert_base_token_classifier_conll03` model and check the results. <br/>

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier = BertForTokenClassification \
    .pretrained('bert_base_token_classifier_conll03', 'en') \
    .setInputCols(['token', 'document']) \
    .setOutputCol('ner') \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('entities')

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    tokenClassifier,
    ner_converter
])

example = spark.createDataFrame([['My name is John Parker! I live in New York and I am a member of the New York Road Runners.']]).toDF("text")
model = pipeline.fit(example)
result= model.transform(example)

# COMMAND ----------

model.stages

# COMMAND ----------

tokenClassifier.getClasses()

# COMMAND ----------

result.columns

# COMMAND ----------

result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.ner.result, result.entities.result)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("ner_label"))

result_df.show(50, truncate=100)

# COMMAND ----------

result_df_1= result.select(F.explode(F.arrays_zip(result.entities.result, result.entities.begin, result.entities.end, result.entities.metadata)).alias("col"))\
                   .select(F.expr("col['0']").alias("entities"),
                            F.expr("col['1']").alias("begin"),
                            F.expr("col['2']").alias("end"),
                            F.expr("col['3']['entity']").alias("ner_label"))
result_df_1.show(50, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  BertForTokenClassification By Using LightPipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Now,  we will use the `bert_large_token_classifier_ontonote` model with LightPipeline and fullAnnotate it with sample data.

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier = BertForTokenClassification \
    .pretrained('bert_large_token_classifier_ontonote', 'en') \
    .setInputCols(['token', 'document']) \
    .setOutputCol('ner') \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(512)

# since output column is IOB/IOB2 style, NerConverter can extract entities
ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('entities')

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    tokenClassifier,
    ner_converter
])

empty_df = spark.createDataFrame([['']]).toDF("text")
model = pipeline.fit(example)

# COMMAND ----------

light_model= LightPipeline(model)
light_result= light_model.fullAnnotate("Steven Rothery is the original guitarist and the longest continuous member of the British rock band Marillion.")[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the classes that `bert_large_token_classifier_ontonote` model can predict

# COMMAND ----------

tokenClassifier.getClasses()

# COMMAND ----------

light_result.keys()

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the ner labels of each token

# COMMAND ----------

tokens= []
ner_labels= []

for i, k in list(zip(light_result["token"], light_result["ner"])):
  tokens.append(i.result)
  ner_labels.append(k.result)

result_df= pd.DataFrame({"tokens": tokens, "ner_labels": ner_labels})
result_df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the chunk results

# COMMAND ----------

chunks= []
begin= []
end= []
ner_label= []

for i in light_result["entities"]:
  chunks.append(i.result)
  begin.append(i.begin)
  end.append(i.end)
  ner_label.append(i.metadata["entity"])

result_df= pd.DataFrame({"chunks": chunks, "begin": begin, "end": end, "ner_label": ner_label})
result_df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC # BertForSequenceClassification

# COMMAND ----------

# MAGIC %md
# MAGIC BertForSequenceClassification can load Bert Models with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.
# MAGIC 
# MAGIC Pretrained models can be loaded with `pretrained()` of the companion object.
# MAGIC <br/><br/>
# MAGIC 
# MAGIC ### **Here are Bert Based Sequence Classification models available in Spark NLP**
# MAGIC <br/>
# MAGIC 
# MAGIC 
# MAGIC | title                                                                                                        | name                                                 | language   |
# MAGIC |:-------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------|:-----------|
# MAGIC | BERT Sequence Classification Base - DBpedia14 (bert_base_sequence_classifier_dbpedia_14)                     | bert_base_sequence_classifier_dbpedia_14             | en         |
# MAGIC | BERT Sequence Classification Base - IMDB (bert_base_sequence_classifier_imdb)                                | bert_base_sequence_classifier_imdb                   | en         |
# MAGIC | BERT Sequence Classification Large - IMDB (bert_large_sequence_classifier_imdb)                              | bert_large_sequence_classifier_imdb                  | en         |
# MAGIC | BERT Sequence Classification Multilingual - AlloCine (bert_multilingual_sequence_classifier_allocine)        | bert_multilingual_sequence_classifier_allocine       | fr         |
# MAGIC | BERT Sequence Classification Base - AG News (bert_base_sequence_classifier_ag_news)                          | bert_base_sequence_classifier_ag_news                | en         |
# MAGIC | BERT Sequence Classification - Spanish Emotion Analysis (bert_sequence_classifier_beto_emotion_analysis)     | bert_sequence_classifier_beto_emotion_analysis       | es         |
# MAGIC | BERT Sequence Classification - Spanish Sentiment Analysis (bert_sequence_classifier_beto_sentiment_analysis) | bert_sequence_classifier_beto_sentiment_analysis     | es         |
# MAGIC | BERT Sequence Classification - Detecting Hate Speech (bert_sequence_classifier_dehatebert_mono)              | bert_sequence_classifier_dehatebert_mono             | en         |
# MAGIC | BERT Sequence Classification - Financial Sentiment Analysis (bert_sequence_classifier_finbert)               | bert_sequence_classifier_finbert                     | en         |
# MAGIC | BERT Sequence Classification - Japanese Sentiment (bert_sequence_classifier_japanese_sentiment)              | bert_sequence_classifier_japanese_sentiment          | ja         |
# MAGIC | BERT Sequence Classification Multilingual Sentiment                                                          | bert_sequence_classifier_multilingual_sentiment      | xx         |
# MAGIC | BERT Sequence Classification - Russian Sentiment Analysis (bert_sequence_classifier_rubert_sentiment)        | bert_sequence_classifier_rubert_sentiment            | ru         |
# MAGIC | BERT Sequence Classification - German Sentiment Analysis (bert_sequence_classifier_sentiment)                | bert_sequence_classifier_sentiment                   | de         |
# MAGIC | BERT Sequence Classification - Turkish Sentiment (bert_sequence_classifier_turkish_sentiment)                | bert_sequence_classifier_turkish_sentiment           | tr         |
# MAGIC | Bert for Sequence Classification (Question vs Statement)                                                     | bert_sequence_classifier_question_statement          | en         |
# MAGIC | Bert for Sequence Classification (Clinical Question vs Statement)                                            | bert_sequence_classifier_question_statement_clinical | en         |
# MAGIC | BERT Sequence Classification - Identify Antisemitic texts                                                    | bert_sequence_classifier_antisemitism                | en         |
# MAGIC | BERT Sequence Classification - Detecting Hate Speech (bert_sequence_classifier_hatexplain)                   | bert_sequence_classifier_hatexplain                  | en         |
# MAGIC | BERT Sequence Classification - Identify Trec Data Classes                                                    | bert_sequence_classifier_trec_coarse                 | en         |
# MAGIC | BERT Sequence Classification - Classify into News Categories                                                 | bert_sequence_classifier_age_news                    | en         |
# MAGIC | BERT Sequence Classification - Classify Banking-Related texts                                                | bert_sequence_classifier_banking77                   | en         |
# MAGIC | BERT Sequence Classification - Detect Spam SMS                                                               | bert_sequence_classifier_sms_spam                    | en         |
# MAGIC | BERT Sequence Classifier - Classify the Music Genre                                                          | bert_sequence_classifier_song_lyrics                 | en         |
# MAGIC | DistilBERT Sequence Classification Base - AG News (distilbert_base_sequence_classifier_ag_news)              | distilbert_base_sequence_classifier_ag_news          | en         |
# MAGIC | DistilBERT Sequence Classification - Amazon Polarity (distilbert_base_sequence_classifier_amazon_polarity)   | distilbert_base_sequence_classifier_amazon_polarity  | en         |
# MAGIC | DistilBERT Sequence Classification - IMDB (distilbert_base_sequence_classifier_imdb)                         | distilbert_base_sequence_classifier_imdb             | en         |
# MAGIC | DistilBERT Sequence Classification - Urdu IMDB (distilbert_base_sequence_classifier_imdb)                    | distilbert_base_sequence_classifier_imdb             | ur         |
# MAGIC | DistilBERT Sequence Classification French - AlloCine (distilbert_multilingual_sequence_classifier_allocine)  | distilbert_multilingual_sequence_classifier_allocine | fr         |
# MAGIC | DistilBERT Sequence Classification - Banking77 (distilbert_sequence_classifier_banking77)                    | distilbert_sequence_classifier_banking77             | en         |
# MAGIC | DistilBERT Sequence Classification - Emotion (distilbert_sequence_classifier_emotion)                        | distilbert_sequence_classifier_emotion               | en         |
# MAGIC | DistilBERT Sequence Classification - Industry (distilbert_sequence_classifier_industry)                      | distilbert_sequence_classifier_industry              | en         |
# MAGIC | DistilBERT Sequence Classification - Policy (distilbert_sequence_classifier_policy)                          | distilbert_sequence_classifier_policy                | en         |
# MAGIC | DistilBERT Sequence Classification - SST-2 (distilbert_sequence_classifier_sst2)                             | distilbert_sequence_classifier_sst2                  | en         |
# MAGIC | ALBERT Sequence Classification Base - AG News (albert_base_sequence_classifier_ag_news)                      | albert_base_sequence_classifier_ag_news              | en         |
# MAGIC | ALBERT Sequence Classification Base - IMDB (albert_base_sequence_classifier_imdb)                            | albert_base_sequence_classifier_imdb                 | en         |
# MAGIC | Longformer Sequence Classification Base - AG News (longformer_base_sequence_classifier_ag_news)              | longformer_base_sequence_classifier_ag_news          | en         |
# MAGIC | Longformer Sequence Classification Base - IMDB (longformer_base_sequence_classifier_imdb)                    | longformer_base_sequence_classifier_imdb             | en         |
# MAGIC | RoBERTa Sequence Classification Base - AG News (roberta_base_sequence_classifier_ag_news)                    | roberta_base_sequence_classifier_ag_news             | en         |
# MAGIC | RoBERTa Sequence Classification Base - IMDB (roberta_base_sequence_classifier_imdb)                          | roberta_base_sequence_classifier_imdb                | en         |
# MAGIC | XLM-RoBERTa Sequence Classification Base - AG News (xlm_roberta_base_sequence_classifier_ag_news)            | xlm_roberta_base_sequence_classifier_ag_news         | en         |
# MAGIC | XLM-RoBERTa Sequence Classification Multilingual - AlloCine (xlm_roberta_base_sequence_classifier_allocine)  | xlm_roberta_base_sequence_classifier_allocine        | fr         |
# MAGIC | XLM-RoBERTa Sequence Classification Base - IMDB (xlm_roberta_base_sequence_classifier_imdb)                  | xlm_roberta_base_sequence_classifier_imdb            | en         |
# MAGIC | XLNet Sequence Classification Base - AG News (xlnet_base_sequence_classifier_ag_news)                        | xlnet_base_sequence_classifier_ag_news               | en         |
# MAGIC | XLNet Sequence Classification Base - IMDB (xlnet_base_sequence_classifier_imdb)                              | xlnet_base_sequence_classifier_imdb                  | en         |

# COMMAND ----------

# MAGIC %md
# MAGIC ## BertForSequenceClassification Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's create a Spark NLP Pipeline with `bert_base_sequence_classifier_imdb` model and check the results. 
# MAGIC 
# MAGIC This model is a fine-tuned BERT model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance.
# MAGIC 
# MAGIC This model has been trained to recognize two types of entities: negative (neg), positive (pos)

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
    .pretrained('bert_base_sequence_classifier_imdb', 'en') \
    .setInputCols(['token', 'document']) \
    .setOutputCol('pred_class') \
    .setCaseSensitive(True) \
    .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
])

sample_text= [["I really liked that movie!"], ["The last movie I watched was awful!"]]
sample_df= spark.createDataFrame(sample_text).toDF("text")
model = pipeline.fit(sample_df)
result= model.transform(sample_df)

# COMMAND ----------

model.stages

# COMMAND ----------

sequenceClassifier.getClasses()

# COMMAND ----------

result.columns

# COMMAND ----------

result_df= result.select(F.explode(F.arrays_zip(result.document.result, result.pred_class.result)).alias("col"))\
                 .select(F.expr("col['0']").alias("sentence"),
                         F.expr("col['1']").alias("prediction"))
                  
result_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DistilBertForSequenceClassification By Using LightPipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we will use distilbert_base_sequence_classifier_ag_news model with LightPipeline and fullAnnotate it with sample data.

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = DistilBertForSequenceClassification \
      .pretrained('distilbert_base_sequence_classifier_ag_news', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

empty_data= spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)

# COMMAND ----------

light_model= LightPipeline(model)
light_result= light_model.fullAnnotate("Manchester United forward Cristiano Ronaldo on Saturday made his 181st appearance for Portugal.")[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the classes that distilbert_base_sequence_classifier_ag_news model can predict

# COMMAND ----------

sequenceClassifier.getClasses()

# COMMAND ----------

light_result.keys()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the prediction

# COMMAND ----------

pd.set_option('display.max_colwidth', None)

text= []
pred= []

for i, k in list(zip(light_result["document"], light_result["class"])):
  text.append(i.result)
  pred.append(k.result)

result_df= pd.DataFrame({"text": text, "prediction": pred})
result_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #