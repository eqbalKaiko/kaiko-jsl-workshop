# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Training and Reusing Named Entity Recognition Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Related blogposts and videos:
# MAGIC 
# MAGIC https://towardsdatascience.com/named-entity-recognition-ner-with-bert-in-spark-nlp-874df20d1d77
# MAGIC 
# MAGIC NerDL worksghop (90 min): https://www.youtube.com/watch?v=YM-e4eOiQ34
# MAGIC 
# MAGIC https://medium.com/spark-nlp/named-entity-recognition-for-healthcare-with-sparknlp-nerdl-and-nercrf-a7751b6ad571
# MAGIC 
# MAGIC https://medium.com/atlas-research/ner-for-clinical-text-7c73caddd180

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ### CoNLL Data Prep

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/src/test/resources/conll2003/eng.train
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/src/test/resources/conll2003/eng.testa

# COMMAND ----------

with open("eng.train") as f:
    train_txt =f.read()

print (train_txt[:500])

# COMMAND ----------

from sparknlp.training import CoNLL

training_data = CoNLL().readDataset(spark, 'file:/databricks/driver/eng.train')

training_data.show(3)

# COMMAND ----------

training_data.printSchema()

# COMMAND ----------

# MAGIC %time training_data.count()

# COMMAND ----------

import pyspark.sql.functions as F

training_data.select(F.explode(F.arrays_zip(training_data.token.result, 
                                            training_data.pos.result, 
                                            training_data.label.result)).alias("cols")) \
             .select(F.expr("cols['0']").alias("token"),
                     F.expr("cols['1']").alias("pos"),
                     F.expr("cols['2']").alias("ner_label")).show(truncate=50)

# COMMAND ----------

training_data.select(F.explode(F.arrays_zip(training_data.token.result,
                                            training_data.label.result)).alias("cols")) \
             .select(F.expr("cols['0']").alias("token"),
                     F.expr("cols['1']").alias("ground_truth")).groupBy('ground_truth').count().orderBy('count', ascending=False).show(100,truncate=False)

# COMMAND ----------

# You can use any word embeddings you want (Glove, Elmo, Bert, custom etc.)

glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
          .setInputCols(["document", "token"])\
          .setOutputCol("embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Graph
# MAGIC 
# MAGIC We will use `TFNerDLGraphBuilder` annotator to create a graph in the model training pipeline. This annotator inspects the data and creates the proper graph if a suitable version of TensorFlow (<= 2.7 ) is available. The graph is stored in the defined folder and loaded by the approach.
# MAGIC 
# MAGIC **ATTENTION:** Do not forget to play with the parameters of this annotator, it may affect the model performance that you want to train.

# COMMAND ----------

# MAGIC %md
# MAGIC **Licensed users** would use this module to create **custom graphs** for each DL model (`ner_dl`, `generic_classifier`, `assertion_dl`, `relation_extraction`) in Spark NLP.

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/ner_logs

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/ner_graphs

# COMMAND ----------

graph_folder = "/dbfs/ner_graphs"

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
              .setOutputLogsPath('dbfs:/ner_logs') # if not set, logs will be written to ~/annotator_logs
          #   .setEnableMemoryOptimizer(True) # if not set, logs will be written to ~/annotator_logs
    
ner_pipeline = Pipeline(stages=[glove_embeddings,
                                graph_builder,
                                nerTagger])


# COMMAND ----------

# MAGIC %md
# MAGIC ### Fitting

# COMMAND ----------

ner_model = ner_pipeline.fit(training_data)

# 1 epoch takes around 1.5 min with batch size=32
# if you get an error for incompatible TF graph, use 4.1 NerDL-Graph.ipynb notebook to create a graph (or see the bottom cell of this notebook)

# COMMAND ----------

# MAGIC %sh cd /dbfs/ner_logs && ls

# COMMAND ----------

# MAGIC %sh head -n 45 /dbfs/ner_logs/NerDLApproach_*

# COMMAND ----------

from sparknlp.training import CoNLL

test_data = CoNLL().readDataset(spark, 'file:/databricks/driver/eng.testa')

test_data = glove_embeddings.transform(test_data)

test_data.show(3)

# COMMAND ----------

predictions = ner_model.transform(test_data)
predictions.show(3)

# COMMAND ----------

predictions.select('token.result','label.result','ner.result').show(3, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test set evaluation

# COMMAND ----------

import pyspark.sql.functions as F

predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                          predictions.label.result,
                                          predictions.ner.result)).alias("cols")) \
                              .select(F.expr("cols['0']").alias("token"),
                                      F.expr("cols['1']").alias("ground_truth"),
                                      F.expr("cols['2']").alias("prediction")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Licensed user will have an access to internal NERDLMetrics module to do this more efficient and easily without going out of Spark. But open source users need to use sklearen.mnetrics or any other equivalent module to do the same.

# COMMAND ----------

from sklearn.metrics import classification_report

preds_df = predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                                     predictions.label.result,
                                                     predictions.ner.result)).alias("cols")) \
                              .select(F.expr("cols['0']").alias("token"),
                                      F.expr("cols['1']").alias("ground_truth"),
                                      F.expr("cols['2']").alias("prediction")).toPandas()

print (classification_report(preds_df['ground_truth'], preds_df['prediction']))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Entity level evaluation (strict eval)

# COMMAND ----------

!wget  -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/utils/conll_eval.py
  
import sys

# Add the path to system, local or mounted S3 bucket, e.g. /dbfs/mnt/<path_to_bucket>
sys.path.append('/databricks/driver/')
#sys.path.append('/databricks/driver/databricks_import_python_module/')
sys.path.append('/databricks/driver/conll_eval.py')

# COMMAND ----------

import conll_eval

metrics = conll_eval.evaluate(preds_df['ground_truth'].values, preds_df['prediction'].values)

# COMMAND ----------

# micro, macro, avg
metrics[0]

# COMMAND ----------

import pandas as pd
pd.DataFrame(metrics[1], columns=['entity','precision','recall','f1','support'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Splitting dataset into train and test

# COMMAND ----------

# MAGIC %md
# MAGIC Also we will use .setTestDataset('ner_dl_test.parquet') for checking test-loss values of each epoch in the logs file and .useBestModel(True) parameter whether to restore and use the model that has achieved the best performance at the end of the training. .

# COMMAND ----------

from sparknlp.training import CoNLL

conll_data = CoNLL().readDataset(spark, 'file:/databricks/driver/eng.train')

(training_data, test_data) = conll_data.randomSplit([0.7, 0.3], seed = 100)

print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save our `test_data` as parquet by transforming with embeddings.

# COMMAND ----------

glove_embeddings.transform(test_data).write.mode("overwrite").parquet('dbfs/nerdl_test.parquet')

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/ner_logs_best

# COMMAND ----------

# MAGIC %md
# MAGIC Let's use `setUseBestModel(True)` parameter to restore the model with the best performance at the end of the training and use the `setTestDataset` parameter to calculate statistical measures for each epoch during training

# COMMAND ----------

nerTagger = NerDLApproach()\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setLabelColumn("label")\
  .setOutputCol("ner")\
  .setMaxEpochs(5)\
  .setLr(0.003)\
  .setBatchSize(32)\
  .setRandomSeed(0)\
  .setVerbose(1)\
  .setValidationSplit(0.2)\
  .setEvaluationLogExtended(True) \
  .setEnableOutputLogs(True)\
  .setIncludeConfidence(True)\
  .setUseBestModel(True)\
  .setGraphFolder(graph_folder)\
  .setTestDataset('dbfs:/nerdl_test.parquet')\
  .setOutputLogsPath('dbfs:/ner_logs_best') # if not set, logs will be written to ~/annotator_logs

ner_pipeline = Pipeline(stages=[
          glove_embeddings,
          graph_builder,
          nerTagger
 ])

# COMMAND ----------

# MAGIC %%time
# MAGIC ner_model = ner_pipeline.fit(training_data)

# COMMAND ----------

# MAGIC %sh cat /dbfs/ner_logs_best/NerDLApproach_*.log

# COMMAND ----------

test_data = glove_embeddings.transform(test_data)

predictions = ner_model.transform(test_data)

from sklearn.metrics import classification_report

preds_df = predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                                     predictions.label.result,
                                                     predictions.ner.result)).alias("cols")) \
                      .select(F.expr("cols['0']").alias("token"),
                              F.expr("cols['1']").alias("ground_truth"),
                              F.expr("cols['2']").alias("prediction")).toPandas()

print (classification_report(preds_df['ground_truth'], preds_df['prediction'], digits=4))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Ner log parser

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/utils/ner_log_parser.py
  
sys.path.append('/databricks/driver/ner_log_parser.py')

# COMMAND ----------

import ner_log_parser

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

# MAGIC %sh cd /dbfs/ner_logs_best && pwd && ls -l

# COMMAND ----------

import os
log_files = os.listdir("/dbfs/ner_logs_best/")
log_files

# COMMAND ----------

ner_log_parser.get_charts('/dbfs/ner_logs_best/'+log_files[0])

# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting Loss**

# COMMAND ----------

ner_log_parser.loss_plot('/dbfs/ner_logs_best/'+log_files[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving the trained model

# COMMAND ----------

ner_model.stages

# COMMAND ----------

# MAGIC %sh cd /databricks/driver/ && ls -la

# COMMAND ----------

ner_model.stages[-1].write().overwrite().save('dbfs:/databricks/driver/models/NerDLModel_5e32b')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Pipeline

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
    
loaded_ner_model = NerDLModel.load("dbfs:/databricks/driver/models/NerDLModel_5e32b")\
     .setInputCols(["sentence", "token", "embeddings"])\
     .setOutputCol("ner")

converter = NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_span")

ner_prediction_pipeline = Pipeline(
    stages = [
        document,
        sentence,
        token,
        glove_embeddings,
        loaded_ner_model,
        converter])

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

from sparknlp.base import LightPipeline

light_model = LightPipeline(prediction_model)

# COMMAND ----------

text = "Peter Parker is a nice guy and lives in New York."

result = light_model.annotate(text)

list(zip(result['token'], result['ner']))

# COMMAND ----------

import pandas as pd

result = light_model.fullAnnotate(text)

ner_df= pd.DataFrame([(int(x.metadata['sentence']), x.result, x.begin, x.end, y.result) for x,y in zip(result[0]["token"], result[0]["ner"])], 
                      columns=['sent_id','token','start','end','ner'])
ner_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Creating your own CoNLL dataset
# MAGIC 
# MAGIC for a detailed overview of how to create a CoNLL file from any annotation, see here >> https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.3.prepare_CoNLL_from_annotations_for_NER.ipynb

# COMMAND ----------

import json
import os
from pyspark.ml import Pipeline
from sparknlp.base import *
from sparknlp.annotator import *
import sparknlp

spark = sparknlp.start()

def get_ann_pipeline ():
    
    document_assembler = DocumentAssembler() \
        .setInputCol("text")\
        .setOutputCol('document')

    sentence = SentenceDetector()\
        .setInputCols(['document'])\
        .setOutputCol('sentence')
    
    tokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")

    pos = PerceptronModel.pretrained() \
              .setInputCols(["sentence", "token"]) \
              .setOutputCol("pos")
    
    embeddings = WordEmbeddingsModel.pretrained()\
          .setInputCols(["sentence", "token"])\
          .setOutputCol("embeddings")

    ner_model = NerDLModel.pretrained() \
          .setInputCols(["sentence", "token", "embeddings"]) \
          .setOutputCol("ner")

    ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")

    ner_pipeline = Pipeline(
        stages = [
            document_assembler,
            sentence,
            tokenizer,
            pos,
            embeddings,
            ner_model,
            ner_converter
        ]
    )

    empty_data = spark.createDataFrame([[""]]).toDF("text")

    ner_pipelineFit = ner_pipeline.fit(empty_data)

    ner_lp_pipeline = LightPipeline(ner_pipelineFit)

    print ("Spark NLP NER lightpipeline is created")

    return ner_lp_pipeline


conll_pipeline = get_ann_pipeline()



# COMMAND ----------

sentences = ["Peter Parker is a nice guy and lives in New York.",
"He is also helping people around the world."]

conll_lines=''

for sentence in sentences:

  parsed = conll_pipeline.annotate (sentence)

  for token, pos, ner in zip(parsed['token'],parsed['pos'],parsed['ner']):

      conll_lines += "{} {} {} {}\n".format(token, pos, pos, ner)

  conll_lines += '\n'


print(conll_lines)

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #