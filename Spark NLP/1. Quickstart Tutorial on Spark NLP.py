# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.Quickstart Tutorial on Spark NLP - 1 hr
# MAGIC 
# MAGIC This is the 1 hr workshop version of the entire training notebooks : https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public

# COMMAND ----------

# MAGIC %md
# MAGIC an intro article for Spark NLP:
# MAGIC 
# MAGIC https://towardsdatascience.com/introduction-to-spark-nlp-foundations-and-basic-components-part-i-c83b7629ed59
# MAGIC 
# MAGIC How to start Spark NLP in 2 weeks:
# MAGIC 
# MAGIC https://towardsdatascience.com/how-to-get-started-with-sparknlp-in-2-weeks-cb47b2ba994d
# MAGIC 
# MAGIC https://towardsdatascience.com/how-to-wrap-your-head-around-spark-nlp-a6f6a968b7e8
# MAGIC 
# MAGIC Article for NER and text classification in Spark NLP
# MAGIC 
# MAGIC https://towardsdatascience.com/named-entity-recognition-ner-with-bert-in-spark-nlp-874df20d1d77
# MAGIC 
# MAGIC https://medium.com/spark-nlp/named-entity-recognition-for-healthcare-with-sparknlp-nerdl-and-nercrf-a7751b6ad571
# MAGIC 
# MAGIC https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32
# MAGIC 
# MAGIC a webinar to show how to train a NER model from scratch (90 min)
# MAGIC 
# MAGIC https://www.youtube.com/watch?v=djWX0MR2Ooo
# MAGIC 
# MAGIC workshop repo that you can start playing with Spark NLP in Colab:
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings
# MAGIC 
# MAGIC Databrikcs Notebooks: 
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/products/databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Coding ...

# COMMAND ----------

import sparknlp

from sparknlp.base import *
from sparknlp.annotator import *

from pyspark.ml import Pipeline

print("Spark NLP version", sparknlp.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Pretrained Pipelines
# MAGIC 
# MAGIC for a more detailed notebook, see https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/1.SparkNLP_Basics.ipynb

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

pipeline_dl = PretrainedPipeline('explain_document_dl', lang='en')


# COMMAND ----------

# MAGIC %md
# MAGIC **Stages**
# MAGIC - DocumentAssembler
# MAGIC - SentenceDetector
# MAGIC - Tokenizer
# MAGIC - NER (NER with GloVe 100D embeddings, CoNLL2003 dataset)
# MAGIC - Lemmatizer
# MAGIC - Stemmer
# MAGIC - Part of Speech
# MAGIC - SpellChecker (Norvig)

# COMMAND ----------

testDoc = '''
Peter Parker is a very good persn.
My life in Russia is very intersting.
John and Peter are brthers. However they don't support each other that much.
Mercedes Benz is also working on a driverless car.
Europe is very culture rich. There are huge churches! and big houses!
'''

result = pipeline_dl.annotate(testDoc)


# COMMAND ----------

result.keys()

# COMMAND ----------

result['entities']

# COMMAND ----------

import pandas as pd

df = pd.DataFrame({'token':result['token'], 'ner_label':result['ner'],
                      'spell_corrected':result['checked'], 'POS':result['pos'],
                      'lemmas':result['lemma'], 'stems':result['stem']})

df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using fullAnnotate to get more details

# COMMAND ----------

detailed_result = pipeline_dl.fullAnnotate(testDoc)

detailed_result[0]['entities']

# COMMAND ----------

chunks=[]
entities=[]
for n in detailed_result[0]['entities']:
        
  chunks.append(n.result)
  entities.append(n.metadata['entity']) 
    
df = pd.DataFrame({'chunks':chunks, 'entities':entities})
df    

# COMMAND ----------

tuples = []

for x,y,z in zip(detailed_result[0]["token"], detailed_result[0]["pos"], detailed_result[0]["ner"]):

  tuples.append((int(x.metadata['sentence']), x.result, x.begin, x.end, y.result, z.result))

df = pd.DataFrame(tuples, columns=['sent_id','token','start','end','pos', 'ner'])

df


# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentiment Analysis

# COMMAND ----------

sentiment = PretrainedPipeline('analyze_sentiment', lang='en')

# COMMAND ----------

result = sentiment.annotate("The movie I watched today was not a good one")

result['sentiment']

# COMMAND ----------

sentiment_imdb_glove = PretrainedPipeline('analyze_sentimentdl_glove_imdb', lang='en')

# COMMAND ----------

comment = '''
It's a very scary film but what impressed me was how true the film sticks to the original's tricks; it isn't filled with loud in-your-face jump scares, in fact, a lot of what makes this film scary is the slick cinematography and intricate shadow play. The use of lighting and creation of atmosphere is what makes this film so tense, which is why it's perfectly suited for those who like Horror movies but without the obnoxious gore.
'''
result = sentiment_imdb_glove.annotate(comment)

result['sentiment']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the modules in a pipeline for custom tasks
# MAGIC 
# MAGIC for a more detailed notebook, see https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/annotation/english/spark-nlp-basics/sample-sentences-en.txt
  
dbutils.fs.cp("file:/databricks/driver/sample-sentences-en.txt", "dbfs:/") 

# COMMAND ----------

with open('sample-sentences-en.txt') as f:
  print (f.read())

# COMMAND ----------

spark_df = spark.read.text('/sample-sentences-en.txt').toDF('text')

spark_df.show(truncate=False)

# COMMAND ----------

textFiles = spark.sparkContext.wholeTextFiles("/sample-sentences-en.txt",4) # or/*.txt
    
spark_df_folder = textFiles.toDF(schema=['path','text'])

spark_df_folder.show(truncate=30)

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     sentenceDetector,
     tokenizer
 ])

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

# COMMAND ----------

result = pipelineModel.transform(spark_df)


# COMMAND ----------

result.show(truncate=20)


# COMMAND ----------

result.printSchema()


# COMMAND ----------

result.select('sentences.result').take(3)


# COMMAND ----------

# MAGIC %md
# MAGIC ### StopWords Cleaner

# COMMAND ----------

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("token")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# COMMAND ----------

stopwords_cleaner.getStopWords()[:10]

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     stopwords_cleaner
 ])

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

result = pipelineModel.transform(spark_df)

result.show()

# COMMAND ----------

result.select('cleanTokens.result').take(1)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Matcher

# COMMAND ----------

! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/news_category_train.csv
  
dbutils.fs.cp("file:/databricks/driver/news_category_train.csv", "dbfs:/")

# COMMAND ----------

news_df = spark.read \
      .option("header", True) \
      .csv("/news_category_train.csv")


news_df.show(5, truncate=50)

# COMMAND ----------

entities = ['Wall Street', 'USD', 'stock', 'NYSE']
with open ('financial_entities.txt', 'w') as f:
    for i in entities:
        f.write(i+'\n')


entities = ['soccer', 'world cup', 'Messi', 'FC Barcelona']
with open ('sport_entities.txt', 'w') as f:
    for i in entities:
        f.write(i+'\n')


# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/financial_entities.txt", "dbfs:/")
dbutils.fs.cp("file:/databricks/driver/sport_entities.txt", "dbfs:/")

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

financial_entity_extractor = TextMatcher() \
    .setInputCols(["document",'token'])\
    .setOutputCol("financial_entities")\
    .setEntities("file:/databricks/driver/financial_entities.txt")\
    .setCaseSensitive(False)\
    .setEntityValue('financial_entity')

sport_entity_extractor = TextMatcher() \
    .setInputCols(["document",'token'])\
    .setOutputCol("sport_entities")\
    .setEntities("file:/databricks/driver/sport_entities.txt")\
    .setCaseSensitive(False)\
    .setEntityValue('sport_entity')


nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     financial_entity_extractor,
     sport_entity_extractor
 ])

empty_df = spark.createDataFrame([['']]).toDF("description")

pipelineModel = nlpPipeline.fit(empty_df)

# COMMAND ----------

result = pipelineModel.transform(news_df)


# COMMAND ----------

result.select('financial_entities.result','sport_entities.result').take(2)


# COMMAND ----------

# MAGIC %md
# MAGIC This means there are no financial and sport entities in the first two lines.

# COMMAND ----------

from pyspark.sql import functions as F

result.select('description','financial_entities.result','sport_entities.result')\
      .toDF('text','financial_matches','sport_matches')\
      .filter((F.size('financial_matches')>1) | (F.size('sport_matches')>1))\
      .show(truncate=70)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Using the pipeline in a LightPipeline

# COMMAND ----------

light_model = LightPipeline(pipelineModel)

light_result = light_model.fullAnnotate("Google, Inc. significantly cut the expected share price for its stock at Wall Street")

light_result[0]['financial_entities']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pretrained Models

# COMMAND ----------

# MAGIC %md
# MAGIC Spark NLP offers the following pre-trained models in around **200+ languages** and all you need to do is to load the pre-trained model into your disk by specifying the model name and then configuring the model parameters as per your use case and dataset. Then you will not need to worry about training a new model from scratch and will be able to enjoy the pre-trained SOTA algorithms directly applied to your own data with transform().
# MAGIC 
# MAGIC In the official documentation, you can find detailed information regarding how these models are trained by using which algorithms and datasets.
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-models

# COMMAND ----------

# MAGIC %md
# MAGIC for a more detailed notebook, see https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC ### LemmatizerModel and ContextSpellCheckerModel

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

spellModel = ContextSpellCheckerModel\
    .pretrained('spellcheck_dl')\
    .setInputCols("token")\
    .setOutputCol("checked")

lemmatizer = LemmatizerModel.pretrained('lemma_antbnc', 'en') \
    .setInputCols(["checked"]) \
    .setOutputCol("lemma")

pipeline = Pipeline(stages = [
    documentAssembler,
    tokenizer,
    spellModel,
    lemmatizer
  ])

empty_ds = spark.createDataFrame([[""]]).toDF("text")

sc_model = pipeline.fit(empty_ds)

lp = LightPipeline(sc_model)

# COMMAND ----------

result = lp.annotate("Plaese alliow me tao introdduce myhelf, I am a man of waelth und tiaste and he just knows that")

list(zip(result['token'],result['checked'],result['lemma']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Word and Sentence Embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC #### Word Embeddings

# COMMAND ----------

glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
      .setInputCols(["document", "token"])\
      .setOutputCol("embeddings")

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     glove_embeddings
 ])

empty_df = spark.createDataFrame([['']]).toDF("description")
pipelineModel = nlpPipeline.fit(empty_df)

result = pipelineModel.transform(news_df.limit(1))

output = result.select('token.result','embeddings.embeddings').limit(1).rdd.flatMap(lambda x: x).collect()


# COMMAND ----------

pd.DataFrame({'token':output[0],'embeddings':output[1]})

# COMMAND ----------

result = pipelineModel.transform(news_df.limit(10))

result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.embeddings.embeddings)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("embeddings"))

result_df.show(10, truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Bert Embeddings

# COMMAND ----------

bert_embeddings = BertEmbeddings.pretrained('bert_base_cased')\
      .setInputCols(["document", "token"])\
      .setOutputCol("embeddings")

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

 
nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     bert_embeddings
 ])

empty_df = spark.createDataFrame([['']]).toDF("description")

pipelineModel = nlpPipeline.fit(empty_df)

result = pipelineModel.transform(news_df.limit(10))

result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.embeddings.embeddings)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("bert_embeddings"))

result_df.show(truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Bert Sentence Embeddings

# COMMAND ----------

bert_sentence_embeddings = BertSentenceEmbeddings.pretrained('sent_small_bert_L6_128')\
    .setInputCols(["document"])\
    .setOutputCol("bert_sent_embeddings")


nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     bert_sentence_embeddings
 ])


empty_df = spark.createDataFrame([['']]).toDF("description")

pipelineModel = nlpPipeline.fit(empty_df)

result = pipelineModel.transform(news_df.limit(10))

result_df = result.select(F.explode(F.arrays_zip(result.document.result, result.bert_sent_embeddings.embeddings)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("document"),
                          F.expr("cols['1']").alias("bert_sent_embeddings"))

result_df.show(truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Universal Sentence Encoder

# COMMAND ----------

# no need for token columns 
use_embeddings = UniversalSentenceEncoder.pretrained('tfhub_use')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

# COMMAND ----------

from pyspark.sql import functions as F

documentAssembler = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     use_embeddings
   ])

empty_df = spark.createDataFrame([['']]).toDF("description")

pipelineModel = nlpPipeline.fit(empty_df)

result = pipelineModel.transform(news_df.limit(10))

result_df = result.select(F.explode(F.arrays_zip(result.document.result, result.sentence_embeddings.embeddings)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("document"),
                          F.expr("cols['1']").alias("USE_embeddings"))

result_df.show(truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Named Entity Recognition (NER) Models

# COMMAND ----------

# MAGIC %md
# MAGIC for a detailed notebbok, see https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/4.NERDL_Training.ipynb

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
      .setInputCols(["document", "token"])\
      .setOutputCol("embeddings")

onto_ner = NerDLModel.pretrained("onto_100", 'en') \
      .setInputCols(["document", "token", "embeddings"]) \
      .setOutputCol("ner")

ner_converter = NerConverter() \
      .setInputCols(["document", "token", "ner"]) \
      .setOutputCol("ner_chunk")


nlpPipeline = Pipeline(stages=[
       documentAssembler, 
       tokenizer,
       glove_embeddings,
       onto_ner,
       ner_converter
 ])

empty_df = spark.createDataFrame([['']]).toDF("description")

pipelineModel = nlpPipeline.fit(empty_df)

# COMMAND ----------

result = pipelineModel.transform(news_df.limit(10))

result.select(F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label")).show(truncate=False)

# COMMAND ----------

light_model = LightPipeline(pipelineModel)

light_result = light_model.fullAnnotate('Peter Parker is a nice persn and lives in New York. Bruce Wayne is also a nice guy and lives in Gotham City.')


chunks = []
entities = []

for n in light_result[0]['ner_chunk']:
        
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    
    
import pandas as pd

df = pd.DataFrame({'chunks':chunks, 'entities':entities})

df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train a NER model

# COMMAND ----------

# MAGIC %md
# MAGIC **To train a new NER from scratch, check out**
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/4.NERDL_Training.ipynb

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/src/test/resources/conll2003/eng.train

#dbutils.fs.cp("file:/databricks/driver/sample-sentences-en.txt", "dbfs:/")

# COMMAND ----------

from sparknlp.training import CoNLL

training_data = CoNLL().readDataset(spark, 'file:/databricks/driver/eng.train')

training_data.show(3)

# COMMAND ----------

training_data.select(F.explode(F.arrays_zip(training_data.token.result, training_data.label.result)).alias("cols"))\
              .select(F.expr("cols['0']").alias("token"),
                      F.expr("cols['1']").alias("ground_truth"))\
              .groupBy('ground_truth').count().orderBy('count', ascending=False).show(100,truncate=False)

# COMMAND ----------

# You can use any word embeddings you want (Glove, Elmo, Bert, custom etc.)

glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
          .setInputCols(["document", "token"])\
          .setOutputCol("embeddings")

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/ner_logs

# COMMAND ----------

nerTagger = NerDLApproach()\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setLabelColumn("label")\
  .setOutputCol("ner")\
  .setMaxEpochs(2)\
  .setLr(0.003)\
  .setPo(0.05)\
  .setBatchSize(32)\
  .setRandomSeed(0)\
  .setVerbose(1)\
  .setValidationSplit(0.2)\
  .setEvaluationLogExtended(True) \
  .setEnableOutputLogs(True)\
  .setIncludeConfidence(True)\
  .setOutputLogsPath('dbfs:/ner_logs') # if not set, logs will be written to ~/annotator_logs
 #.setGraphFolder('graphs') >> put your graph file (pb) under this folder if you are using a custom graph generated thru NerDL-Graph
    
    
ner_pipeline = Pipeline(stages=[
          glove_embeddings,
          nerTagger
 ])

# COMMAND ----------

# remove the existing logs

!rm -r /dbfs/ner_logs/*

# COMMAND ----------

ner_model = ner_pipeline.fit(training_data)

# 1 epoch takes around 2.5 min with batch size=32
# if you get an error for incompatible TF graph, use NERDL Graph script to generate the necessary TF graph at the end of this notebook

# COMMAND ----------

# MAGIC %sh cd /dbfs/ner_logs && pwd && ls -l

# COMMAND ----------

# MAGIC %sh head -n 45 /dbfs/ner_logs/NerDLApproach_*

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/models

# COMMAND ----------

ner_model.stages[1].write().overwrite().save('dbfs:/models/NER_glove_e2_b32')

# COMMAND ----------

# MAGIC %sh cd /dbfs/models/ && pwd && ls -l

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load saved model

# COMMAND ----------

document = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentence = SentenceDetector()\
        .setInputCols(['document'])\
        .setOutputCol('sentence')

token = Tokenizer()\
        .setInputCols(['sentence'])\
        .setOutputCol('token')

glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
        .setInputCols(["document", "token"])\
        .setOutputCol("embeddings")
  
# load back and use in any pipeline
loaded_ner_model = NerDLModel.load("dbfs:/models/NER_glove_e2_b32")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

converter = NerConverter()\
        .setInputCols(["document", "token", "ner"])\
        .setOutputCol("ner_span")

ner_prediction_pipeline = Pipeline(stages = [
        document,
        sentence,
        token,
        glove_embeddings,
        loaded_ner_model,
        converter
])

# COMMAND ----------

empty_data = spark.createDataFrame([['']]).toDF("text")

prediction_model = ner_prediction_pipeline.fit(empty_data)


# COMMAND ----------

text = "Peter Parker is a nice guy and lives in New York."

sample_data = spark.createDataFrame([[text]]).toDF("text")

sample_data.show(truncate=False)

# COMMAND ----------

preds = prediction_model.transform(sample_data)

preds.select(F.explode(F.arrays_zip(preds.ner_span.result,preds.ner_span.metadata)).alias("entities")) \
      .select(F.expr("entities['0']").alias("chunk"),
              F.expr("entities['1'].entity").alias("entity")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train a model with Graph Builder

# COMMAND ----------

# MAGIC %md
# MAGIC We will use `TFNerDLGraphBuilder` annotator to create a graph in the model training pipeline. This annotator inspects the data and creates the proper graph if a suitable version of TensorFlow (<= 2.7 ) is available. The graph is stored in the defined folder and loaded by the approach.
# MAGIC 
# MAGIC **ATTENTION:** Do not forget to play with the parameters of this annotator, it may affect the model performance that you want to train.

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/ner_logs_custom_graph

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/custom_ner_graphs

# COMMAND ----------

graph_folder = "dbfs:/custom_ner_graphs"

# COMMAND ----------

graph_builder = TFNerDLGraphBuilder()\
              .setInputCols(["sentence", "token", "embeddings"]) \
              .setLabelColumn("label")\
              .setGraphFile("auto")\
              .setGraphFolder(graph_folder)\
              .setHiddenUnitsNumber(20)

# COMMAND ----------

nerTagger = NerDLApproach()\
              .setInputCols(["sentence", "token", "embeddings"])\
              .setLabelColumn("label")\
              .setOutputCol("ner")\
              .setMaxEpochs(3)\
              .setLr(0.003)\
              .setBatchSize(32)\
              .setRandomSeed(0)\
              .setVerbose(1)\
              .setValidationSplit(0.2)\
              .setEvaluationLogExtended(True) \
              .setEnableOutputLogs(True)\
              .setIncludeConfidence(True)\
              .setGraphFolder(graph_folder)\
              .setOutputLogsPath('dbfs:/ner_logs_custom_graph') # if not set, logs will be written to ~/annotator_logs
          #   .setEnableMemoryOptimizer(True) # if not set, logs will be written to ~/annotator_logs
    
ner_pipeline = Pipeline(stages=[glove_embeddings,
                                graph_builder,
                                nerTagger])

# COMMAND ----------

ner_model = ner_pipeline.fit(training_data)

# COMMAND ----------

# MAGIC %sh cd /dbfs/ner_logs_custom_graph && pwd && ls -l

# COMMAND ----------

# MAGIC %sh head -n 45 /dbfs/ner_logs_custom_graph/NerDLApproach_*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Classification
# MAGIC 
# MAGIC for a detailed notebook, see https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb

# COMMAND ----------

! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/news_category_test.csv
  
dbutils.fs.cp("file:/databricks/driver/news_category_test.csv", "dbfs:/") 

# COMMAND ----------

from pyspark.sql.functions import col

trainDataset = spark.read \
      .option("header", True) \
      .csv("/news_category_train.csv")

trainDataset.groupBy("category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# COMMAND ----------

testDataset = spark.read \
      .option("header", True) \
      .csv("/news_category_test.csv")


testDataset.groupBy("category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# COMMAND ----------

# MAGIC %fs mkdirs dbfs:/clf_dl_logs

# COMMAND ----------

# actual content is inside description column
document = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")
    
# we can also use sentece detector here if we want to train on and get predictions for each sentence

use_embeddings = UniversalSentenceEncoder.pretrained('tfhub_use')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
classsifierdl = ClassifierDLApproach()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("class")\
    .setLabelColumn("category")\
    .setMaxEpochs(5)\
    .setBatchSize(8)\
    .setLr(0.001)\
    .setEnableOutputLogs(True)\
    .setOutputLogsPath('dbfs:/clf_dl_logs') 

use_clf_pipeline = Pipeline(
    stages = [
        document,
        use_embeddings,
        classsifierdl
    ])

# COMMAND ----------

# remove the existing logs

! rm -r /dbfs/clf_dl_logs/*

# COMMAND ----------

use_pipelineModel = use_clf_pipeline.fit(trainDataset)
# 5 epochs takes around 3 min

# COMMAND ----------

# MAGIC %sh cd /dbfs/clf_dl_logs/ && ls -lt

# COMMAND ----------

# MAGIC %sh cat  /dbfs/clf_dl_logs/ClassifierDLApproach*

# COMMAND ----------

from sparknlp.base import LightPipeline

light_model = LightPipeline(use_pipelineModel)

text='''
Fearing the fate of Italy, the centre-right government has threatened to be merciless with those who flout tough restrictions. 
As of Wednesday it will also include all shops being closed across Greece, with the exception of supermarkets. Banks, pharmacies, pet-stores, mobile phone stores, opticians, bakers, mini-markets, couriers and food delivery outlets are among the few that will also be allowed to remain open.
'''
result = light_model.annotate(text)

result['class']

# COMMAND ----------

light_model.annotate('the soccer games will be postponed.')


# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #