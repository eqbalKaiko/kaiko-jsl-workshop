# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.Training and Reusing Text Classification Models

# COMMAND ----------

# MAGIC %md
# MAGIC **Relevant blogpost:** https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32

# COMMAND ----------

import json
import os
import string
import pandas as pd
import numpy as np

import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.util import *

import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 100)

print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Pretrained ClassifierDL and SentimentDL models

# COMMAND ----------

fake_classifier = ClassifierDLModel.pretrained('classifierdl_use_fakenews', 'en') \
                .setInputCols(["document", "sentence_embeddings"]) \
                .setOutputCol("class")

# COMMAND ----------

# MAGIC %md
# MAGIC fake_news classifier is trained on `https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv.zip`

# COMMAND ----------

fake_classifier.getClasses()

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained(name="tfhub_use",lang="en") \
      .setInputCols(["document"])\
      .setOutputCol("sentence_embeddings")

nlpPipeline = Pipeline(stages=[
      documentAssembler, 
      use,
      fake_classifier
  ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

fake_clf_model = nlpPipeline.fit(empty_data)


# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/spam_ham_dataset.csv
  
dbutils.fs.cp("file:/databricks/driver/spam_ham_dataset.csv", "dbfs:/") 

# COMMAND ----------

fake_lp_pipeline = LightPipeline(fake_clf_model)

text = 'BREAKING: Leaked Picture Of Obama Being Dragged Before A Judge In Handcuffs For Wiretapping Trump'

fake_lp_pipeline.annotate(text)

# COMMAND ----------

sample_data = spark.createDataFrame([[text]]).toDF("text")

sample_data.show(truncate=False)

# COMMAND ----------

pred = fake_clf_model.transform(sample_data)


# COMMAND ----------

pred.show()

# COMMAND ----------

pred.select('text','class.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC you can find more samples here >> `https://github.com/KaiDMML/FakeNewsNet/tree/master/dataset`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generic classifier function

# COMMAND ----------

def get_clf_lp(model_name, sentiment_dl=False, pretrained=True):

  documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

  use = UniversalSentenceEncoder.pretrained(lang="en") \
      .setInputCols(["document"])\
      .setOutputCol("sentence_embeddings")


  if pretrained:

    if sentiment_dl:

      document_classifier = SentimentDLModel.pretrained(model_name, 'en') \
                .setInputCols(["document", "sentence_embeddings"]) \
                .setOutputCol("class")
    else:
      document_classifier = ClassifierDLModel.pretrained(model_name, 'en') \
                .setInputCols(["document", "sentence_embeddings"]) \
                .setOutputCol("class")

  else:

    if sentiment_dl:

      document_classifier = SentimentDLModel.load(model_name) \
                .setInputCols(["document", "sentence_embeddings"]) \
                .setOutputCol("class")
    else:
      document_classifier = ClassifierDLModel.load(model_name) \
                .setInputCols(["document", "sentence_embeddings"]) \
                .setOutputCol("class")

  print ('classes:',document_classifier.getClasses())

  nlpPipeline = Pipeline(stages=[
                  documentAssembler, 
                  use,
                  document_classifier
  ])

  empty_data = spark.createDataFrame([[""]]).toDF("text")

  clf_pipelineFit = nlpPipeline.fit(empty_data)

  clf_lp_pipeline = LightPipeline(clf_pipelineFit)

  return clf_lp_pipeline


# COMMAND ----------

clf_lp_pipeline = get_clf_lp('classifierdl_use_trec50')

# COMMAND ----------

# MAGIC %md
# MAGIC trained on the TREC datasets:
# MAGIC 
# MAGIC Classify open-domain, fact-based questions into one of the following broad semantic categories: 
# MAGIC 
# MAGIC ```Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values.```

# COMMAND ----------

text = 'What was the number of member nations of the U.N. in 2000?'

clf_lp_pipeline.annotate(text)['class']

# COMMAND ----------

clf_lp_pipeline.fullAnnotate(text)[0]['class'][0].result

# COMMAND ----------

clf_lp_pipeline.fullAnnotate(text)[0]['class'][0].metadata

# COMMAND ----------

text = 'What animal was the first mammal successfully cloned from adult cells?'

clf_lp_pipeline.annotate(text)['class']

# COMMAND ----------

clf_lp_pipeline = get_clf_lp('classifierdl_use_cyberbullying')


# COMMAND ----------

text ='RT @EBeisner @ahall012 I agree with you!! I would rather brush my teeth with sandpaper then watch football with a girl!!'

clf_lp_pipeline.annotate(text)['class']

# COMMAND ----------

clf_lp_pipeline = get_clf_lp('classifierdl_use_fakenews')


# COMMAND ----------

text ='Donald Trump a KGB Spy? 11/02/2016 In today’s video, Christopher Greene of AMTV reports Hillary Clinton campaign accusation that Donald Trump is a KGB spy is about as weak and baseless a claim as a Salem witch hunt or McCarthy era trial. It’s only because Hillary Clinton is losing that she is lobbing conspiracy theory. Citizen Quasar The way I see it, one of two things will happen: 1. Trump will win by a landslide but the election will be stolen via electronic voting, just like I have been predicting for over a decade, and the American People will accept the skewed election results just like they accept the TSA into their crotches. 2. Somebody will bust a cap in Hillary’s @$$ killing her and the election will be postponed. Follow AMTV!'

clf_lp_pipeline.annotate(text)['class']


# COMMAND ----------

text ='Sen. Marco Rubio (R-Fla.) is adding a veteran New Hampshire political operative to his team as he continues mulling a possible 2016 presidential bid, the latest sign that he is seriously preparing to launch a campaign later this year.Jim Merrill, who worked for former GOP presidential nominee Mitt Romney and ran his 2008 and 2012 New Hampshire primary campaigns, joined Rubio’s fledgling campaign on Monday, aides to the senator said.Merrill will be joining Rubio’s Reclaim America PAC to focus on Rubio’s New Hampshire and broader Northeast political operations."Marco has always been well received in New Hampshire, and should he run for president, he would be very competitive there," Terry Sullivan, who runs Reclaim America, said in a statement. "Jim certainly knows how to win in New Hampshire and in the Northeast, and will be a great addition to our team at Reclaim America.”News of Merrill’s hire was first reported by The New York Times.'

clf_lp_pipeline.annotate(text)['class']

# COMMAND ----------

sentiment_lp_pipeline = get_clf_lp('sentimentdl_use_twitter', sentiment_dl=True)

# COMMAND ----------

text ='I am SO happy the news came out in time for my birthday this weekend! My inner 7-year-old cannot WAIT!'

sentiment_lp_pipeline.annotate(text)['class']

# COMMAND ----------

sentiment_lp_pipeline = get_clf_lp('classifierdl_use_emotion', sentiment_dl=False)


# COMMAND ----------

sentiment_lp_pipeline.annotate(text)['class']

# COMMAND ----------

# MAGIC %md
# MAGIC ## ClassiferDL with Word Embeddings and Text Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset

# COMMAND ----------

! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/news_category_train.csv
! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/news_category_test.csv
  
dbutils.fs.cp("file:/databricks/driver/news_category_train.csv", "dbfs:/")
dbutils.fs.cp("file:/databricks/driver/news_category_test.csv", "dbfs:/")

# COMMAND ----------

# MAGIC %sh cd /databricks/driver/ && ls -lt 

# COMMAND ----------

trainDataset = spark.read \
      .option("header", True) \
      .csv("/news_category_train.csv")

trainDataset.show(truncate=50)

# COMMAND ----------

trainDataset.count()


# COMMAND ----------

from pyspark.sql.functions import col

trainDataset.groupBy("category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# COMMAND ----------

testDataset = spark.read \
      .option("header", True) \
      .csv("/news_category_test.csv")

testDataset.show(truncate=50)

# COMMAND ----------

testDataset.groupBy("category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# COMMAND ----------

# if we want to split the dataset
'''(trainData, testData) = trainDataset.randomSplit([0.7, 0.3], seed = 100)
print("Train Dataset Count: " + str(trainData.count()))
print("Test Dataset Count: " + str(testData.count()))'''


# COMMAND ----------

document_assembler = DocumentAssembler() \
      .setInputCol("description") \
      .setOutputCol("document")

tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")
    
normalizer = Normalizer() \
      .setInputCols(["token"]) \
      .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

lemma = LemmatizerModel.pretrained('lemma_antbnc') \
      .setInputCols(["cleanTokens"]) \
      .setOutputCol("lemma")


# COMMAND ----------

# MAGIC %md
# MAGIC ### with Glove 100d embeddings

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/clf_glove_logs

# COMMAND ----------


glove_embeddings = WordEmbeddingsModel.pretrained("glove_100d") \
      .setInputCols(["document",'lemma'])\
      .setOutputCol("embeddings")\
      .setCaseSensitive(False)

embeddingsSentence = SentenceEmbeddings() \
      .setInputCols(["document", "embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

classsifierdl = ClassifierDLApproach()\
      .setInputCols(["sentence_embeddings"])\
      .setOutputCol("class")\
      .setLabelColumn("category")\
      .setMaxEpochs(2)\
      .setBatchSize(64) \
      .setLr(5e-3) \
      .setDropout(0.5)\
      .setEnableOutputLogs(True)\
      .setOutputLogsPath('dbfs:/clf_glove_logs')

clf_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            lemma, 
            glove_embeddings,
            embeddingsSentence,
            classsifierdl])

# COMMAND ----------

'''
default classifierDL params:

maxEpochs -> 10,
lr -> 5e-3f,
dropout -> 0.5f,
batchSize -> 64,
enableOutputLogs -> false,
verbose -> Verbose.Silent.id,
validationSplit -> 0.0f,
outputLogsPath -> ""
'''

# COMMAND ----------

# Train (2 min for 2 epochs)

clf_pipelineModel = clf_pipeline.fit(trainDataset)

# COMMAND ----------

# MAGIC %sh cd /dbfs/clf_glove_logs/ && ls -lt 

# COMMAND ----------

# MAGIC %sh cat /dbfs/clf_glove_logs/* 

# COMMAND ----------

# get the predictions on test Set

preds = clf_pipelineModel.transform(testDataset)

preds.select('category','description',"class.result").show(10, truncate=80)

# COMMAND ----------

# due to bug in cluster mode (https://github.com/JohnSnowLabs/spark-nlp/issues/857) , as a workaround, you can just save the fitted model and then load back from dbfs and then transform on the test set. 
clf_pipelineModel.stages[-1].write().overwrite().save('dbfs:/databricks/driver/models/ClassifierDL_wordemb_g100d')
classsifierdlmodel_loaded = ClassifierDLModel.load('dbfs:/databricks/driver/models/ClassifierDL_wordemb_g100d')


clf_pipeline_pred = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            lemma, 
            glove_embeddings,
            clf_pipelineModel.stages[-2],
            classsifierdlmodel_loaded])

empty_data = spark.createDataFrame([[""]]).toDF("description")

result = clf_pipeline_pred.fit(empty_data).transform(testDataset)

# COMMAND ----------

preds_df = result.select('category','description',"class.result").toPandas()

# The result is an array since in Spark NLP you can have multiple sentences.
# Let's explode the array and get the item(s) inside of result column out
preds_df['result'] = preds_df['result'].apply(lambda x : x[0])

# We are going to use sklearn to evalute the results on test dataset
from sklearn.metrics import classification_report

print (classification_report(preds_df['result'], preds_df['category']))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting prediction from Trained model

# COMMAND ----------

from sparknlp.base import LightPipeline

light_model = LightPipeline(clf_pipelineModel)

# COMMAND ----------

text='''
Fearing the fate of Italy, the centre-right government has threatened to be merciless with those who flout tough restrictions. 
As of Wednesday it will also include all shops being closed across Greece, with the exception of supermarkets. Banks, pharmacies, pet-stores, mobile phone stores, opticians, bakers, mini-markets, couriers and food delivery outlets are among the few that will also be allowed to remain open.
'''
result = light_model.annotate(text)

result['class']

# COMMAND ----------

# MAGIC %md
# MAGIC ## ClassifierDL with Universal Sentence Embeddings

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/clf_use_logs

# COMMAND ----------

# actual content is inside description column
document = DocumentAssembler()\
      .setInputCol("description")\
      .setOutputCol("document")
    
# we can also use sentece detector here if we want to train on and get predictions for each sentence

use = UniversalSentenceEncoder.pretrained()\
     .setInputCols(["document"])\
     .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
classsifierdl = ClassifierDLApproach()\
      .setInputCols(["sentence_embeddings"])\
      .setOutputCol("class")\
      .setLabelColumn("category")\
      .setMaxEpochs(5)\
      .setLr(0.001)\
      .setBatchSize(8)\
      .setEnableOutputLogs(True)\
      .setOutputLogsPath('dbfs:/clf_use_logs')

use_clf_pipeline = Pipeline(
    stages = [
        document,
        use,
        classsifierdl
    ])

# COMMAND ----------

# 5 epochs takes around 8 min

use_pipelineModel = use_clf_pipeline.fit(trainDataset)

# COMMAND ----------

# MAGIC %sh cd  /dbfs/clf_use_logs/ && ls -l

# COMMAND ----------

# MAGIC %sh cat /dbfs/clf_use_logs/*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving & loading back the trained model

# COMMAND ----------

use_pipelineModel.stages

# COMMAND ----------

use_pipelineModel.stages[2].write().overwrite().save('/databricks/driver/models/ClassifierDL_USE_e5')

# COMMAND ----------

classsifierdlmodel = ClassifierDLModel.load('dbfs:/databricks/driver/models/ClassifierDL_USE_e5')

# COMMAND ----------

clf_lp = get_clf_lp('dbfs:/databricks/driver/models/ClassifierDL_USE_e5', sentiment_dl=False,  pretrained=False)

# COMMAND ----------

clf_lp.annotate(text)['class']

# COMMAND ----------

# MAGIC %md
# MAGIC # SentimentDL Classifier

# COMMAND ----------

# MAGIC %md
# MAGIC see also here >> `https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb`

# COMMAND ----------

!wget -q aclimdb_train.csv https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment-corpus/aclimdb/aclimdb_train.csv
!wget -q aclimdb_test.csv https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment-corpus/aclimdb/aclimdb_test.csv
  
dbutils.fs.cp("file:/databricks/driver/aclimdb_train.csv", "dbfs:/")
dbutils.fs.cp("file:/databricks/driver/aclimdb_test.csv", "dbfs:/")

# COMMAND ----------

trainDataset = spark.read \
      .option("header", True) \
      .csv("/aclimdb_train.csv")

trainDataset.show(10)

# COMMAND ----------

testDataset = spark.read \
      .option("header", True) \
      .csv("/aclimdb_test.csv")

testDataset.show(10)

# COMMAND ----------

# actual content is inside description column
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained() \
   .setInputCols(["document"])\
   .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
sentimentdl = SentimentDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("label")\
  .setMaxEpochs(5)\
  .setEnableOutputLogs(True)

pipeline = Pipeline(stages = [
    document,
    use,
    sentimentdl
    ])


# COMMAND ----------

pipelineModel = pipeline.fit(trainDataset)

# COMMAND ----------

result = pipelineModel.transform(testDataset)

result_df = result.select('text','label',"class.result").toPandas()

result_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # MultiLabel Classifier

# COMMAND ----------

# MAGIC %md
# MAGIC see also here >> `https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_toxic_classifier.ipynb`

# COMMAND ----------

!curl -O 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/toxic_comments/toxic_train.snappy.parquet'
!curl -O 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/classifier-dl/toxic_comments/toxic_test.snappy.parquet'

dbutils.fs.cp("file:/databricks/driver/toxic_train.snappy.parquet", "dbfs:/")
dbutils.fs.cp("file:/databricks/driver/toxic_test.snappy.parquet", "dbfs:/")

# COMMAND ----------

trainDataset = spark.read.parquet("/toxic_train.snappy.parquet").repartition(120)
testDataset = spark.read.parquet("/toxic_test.snappy.parquet").repartition(10)

# COMMAND ----------

# Let's use shrink to remove new lines in the comments
document = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")\
  .setCleanupMode("shrink")

# Here we use the state-of-the-art Universal Sentence Encoder model from TF Hub
embeddings = UniversalSentenceEncoder.pretrained() \
  .setInputCols(["document"])\
  .setOutputCol("sentence_embeddings")

# We will use MultiClassifierDL built by using Bidirectional GRU and CNNs inside TensorFlow that supports up to 100 classes
# We will use only 5 Epochs but feel free to increase it on your own dataset
multiClassifier = MultiClassifierDLApproach()\
  .setInputCols("sentence_embeddings")\
  .setOutputCol("category")\
  .setLabelColumn("labels")\
  .setBatchSize(128)\
  .setMaxEpochs(5)\
  .setLr(1e-3)\
  .setThreshold(0.5)\
  .setShufflePerEpoch(False)\
  .setEnableOutputLogs(True)\
  .setValidationSplit(0.1)

pipeline = Pipeline(
    stages = [
        document,
        embeddings,
        multiClassifier
    ])

# COMMAND ----------

pipelineModel = pipeline.fit(trainDataset)

# COMMAND ----------

preds = pipelineModel.transform(testDataset)
preds_df = preds.select('text','labels',"category.result").toPandas()
preds_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Case Study: Conference Title Classification

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/title_conference.csv
  
dbutils.fs.cp("file:/databricks/driver/title_conference.csv", "dbfs:/")

# COMMAND ----------

import pandas as pd
df = pd.read_csv('title_conference.csv')
df

# COMMAND ----------

df.Conference.value_counts()

# COMMAND ----------

trainDataset = spark.read \
      .option("header", True) \
      .csv("/title_conference.csv")

(trainingData, testData) = trainDataset.randomSplit([0.8, 0.2], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))


# COMMAND ----------

document = DocumentAssembler()\
      .setInputCol("Title")\
      .setOutputCol("document")
    
# we can also use sentece detector here if we want to train on and get predictions for each sentence

use = UniversalSentenceEncoder.pretrained()\
      .setInputCols(["document"])\
      .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
classsifierdl = ClassifierDLApproach()\
      .setInputCols(["sentence_embeddings"])\
      .setOutputCol("class")\
      .setLabelColumn("Conference")\
      .setMaxEpochs(20)\
      .setEnableOutputLogs(True)

use_clf_pipeline = Pipeline(
    stages = [
        document,
        use,
        classsifierdl
    ])

# COMMAND ----------

use_pipelineModel = use_clf_pipeline.fit(trainingData)

# 20 epochs takes around 25 seconds !

# COMMAND ----------

use_pipelineModel.stages

# COMMAND ----------

use_pipelineModel.stages[-1].write().overwrite().save('dbfs:/databricks/driver/models/use_clf')
use_clf_loaded = ClassifierDLModel.load('dbfs:/databricks/driver/models/use_clf')

use_clf_pipeline_pred = Pipeline(
    stages=[document, 
            use_pipelineModel.stages[-2],
            use_clf_loaded])

empty_data = spark.createDataFrame([[""]]).toDF("description")

result = use_clf_pipeline_pred.fit(empty_data).transform(testData)

# COMMAND ----------

result.select('Title','Conference',"class.result").show(10, truncate=80)


# COMMAND ----------

# We are going to use sklearn to evalute the results on test dataset
preds_df = result.select('Conference','Title',"class.result").toPandas()

# Let's explode the array and get the item(s) inside of result column out
preds_df['result'] = preds_df['result'].apply(lambda x : x[0])

from sklearn.metrics import classification_report

print (classification_report(preds_df['result'], preds_df['Conference']))


# COMMAND ----------

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

preds = pd.DataFrame(confusion_matrix(preds_df['result'], preds_df['Conference']), columns = np.unique(preds_df['Conference']), index =  np.unique(preds_df['Conference']))
preds

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook # 4