# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC #Zero-Shot Clinical Relation Extraction Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Zero-shot Relation Extraction to extract relations between clinical entities with no training dataset
# MAGIC This release includes a zero-shot relation extraction model that leverages `BertForSequenceClassificaiton` to return, based on a predefined set of relation candidates (including no-relation / O), which one has the higher probability to be linking two entities.
# MAGIC 
# MAGIC The dataset will be a csv which contains the following columns: `sentence`, `chunk1`, `firstCharEnt1`, `lastCharEnt1`, `label1`, `chunk2`, `firstCharEnt2`, `lastCharEnt2`, `label2`, `rel`.

# COMMAND ----------

# MAGIC %md
# MAGIC The relation types (TeRP, TrAP, PIP, TrNAP, etc...) are described [here](https://www.i2b2.org/NLP/Relations/assets/Relation%20Annotation%20Guideline.pdf)
# MAGIC 
# MAGIC Let's take a look at the first sentence!
# MAGIC 
# MAGIC `She states this light-headedness is often associated with shortness of breath and diaphoresis occasionally with nausea`
# MAGIC 
# MAGIC As we see in the table, the sentences includes a `PIP` relationship (`Medical problem indicates medical problem`), meaning that in that sentence, chunk1 (`light-headedness`) *indicates* chunk2 (`diaphoresis`).
# MAGIC 
# MAGIC We set a list of candidates tags (`[PIP, TrAP, TrNAP, TrWP, O]`) and candidate sentences (`[light-headedness caused diaphoresis, light-headedness was administered for diaphoresis, light-headedness was not given for diaphoresis, light-headedness worsened diaphoresis]`), meaning that:
# MAGIC 
# MAGIC - `PIP` is expressed by `light-headedness caused diaphoresis`
# MAGIC - `TrAP` is expressed by `light-headedness was administered for diaphoresis`
# MAGIC - `TrNAP` is expressed by `light-headedness was not given for diaphoresis`
# MAGIC - `TrWP` is expressed by `light-headedness worsened diaphoresis`
# MAGIC - or something generic, like `O` is expressed by `light-headedness and diaphoresis`...
# MAGIC 
# MAGIC We will get that the biggest probability of is `PIP`, since it's phrase `light-headedness caused diaphoresis` is the most similar relationship expressing the meaning in the original sentence (`light-headnedness is often associated with ... and diaphoresis`)

# COMMAND ----------

import sparknlp
import sparknlp_jsl
import pandas as pd
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline,PipelineModel

print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Zero Shot Relation Extraction
# MAGIC Using the pretrained `re_zeroshot_biobert` model, available in Models Hub under the Relation Extraction category.

# COMMAND ----------

# Clinical NER

documenter = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentencer = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_clinical = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_clinical")

ner_clinical_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_clinical"]) \
    .setOutputCol("ner_clinical_chunks")\
    .setWhiteList(["PROBLEM", "TEST"])      # PROBLEM-TEST-TREATMENT

ner_posology = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_posology")           

ner_posology_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_posology"]) \
    .setOutputCol("ner_posology_chunks")\
    .setWhiteList(["DRUG"])                # DRUG-FREQUENCY-DOSAGE-DURATION-FORM-ROUTE-STRENGTH

chunk_merger = ChunkMergeApproach()\
    .setInputCols("ner_clinical_chunks", "ner_posology_chunks")\
    .setOutputCol('merged_ner_chunks')


## ZERO-SHOT RE Starting...

pos_tagger = PerceptronModel() \
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["document", "pos_tags", "tokens"]) \
    .setOutputCol("dependencies")

re_ner_chunk_filter = RENerChunksFilter() \
    .setRelationPairs(["problem-test","problem-drug"]) \
    .setMaxSyntacticDistance(4)\
    .setDocLevelRelations(False)\
    .setInputCols(["merged_ner_chunks", "dependencies"]) \
    .setOutputCol("re_ner_chunks")

re_model = ZeroShotRelationExtractionModel.pretrained("re_zeroshot_biobert", "en", "clinical/models")\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")\
    .setRelationalCategories({
                            "ADE": ["{DRUG} causes {PROBLEM}."],
                            "IMPROVE": ["{DRUG} improves {PROBLEM}.", "{DRUG} cures {PROBLEM}."],
                            "REVEAL": ["{TEST} reveals {PROBLEM}."]})\
    .setMultiLabel(True)

pipeline = sparknlp.base.Pipeline() \
    .setStages([documenter,  
                sentencer,
                tokenizer, 
                words_embedder, 
                ner_clinical, 
                ner_clinical_converter,
                ner_posology, 
                ner_posology_converter,
                chunk_merger,
                pos_tagger, 
                dependency_parser, 
                re_ner_chunk_filter, 
                re_model])

# COMMAND ----------

# create Spark DF

sample_text = "Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer."

data = spark.createDataFrame([[sample_text]]).toDF("text")
data.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Fit the model and transform it with the dataframe.**

# COMMAND ----------

model = pipeline.fit(data)
results = model.transform(data)

# COMMAND ----------

# relations output

results.selectExpr("explode(relations) as relation").show(truncate=False)

# COMMAND ----------

# results in Spark DF 

from pyspark.sql import functions as F

results.select(F.explode(F.arrays_zip(results.relations.metadata, results.relations.result)).alias("cols"))\
       .select(F.expr("cols['0']['sentence']").alias("sentence"),
               F.expr("cols['0']['entity1_begin']").alias("entity1_begin"),
               F.expr("cols['0']['entity1_end']").alias("entity1_end"),
               F.expr("cols['0']['chunk1']").alias("chunk1"),
               F.expr("cols['0']['entity1']").alias("entity1"),
               F.expr("cols['0']['entity2_begin']").alias("entity2_begin"),
               F.expr("cols['0']['entity2_end']").alias("entity2_end"),
               F.expr("cols['0']['chunk2']").alias("chunk2"),
               F.expr("cols['0']['entity2']").alias("entity2"),
               F.expr("cols['0']['hypothesis']").alias("hypothesis"),
               F.expr("cols['0']['nli_prediction']").alias("nli_prediction"),
               F.expr("cols['1']").alias("relation"),
               F.expr("cols['0']['confidence']").alias("confidence"),
       ).show(truncate=70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LightPipeline
# MAGIC 
# MAGIC LightPipelines are Spark NLP specific Pipelines, equivalent to Spark ML Pipeline, but meant to deal with smaller amounts of data. They’re useful working with small datasets, debugging results, or when running either training or prediction from an API that serves one-off requests.
# MAGIC Spark NLP LightPipelines are Spark ML pipelines converted into a single machine but the multi-threaded task, becoming more than 10x times faster for smaller amounts of data (small is relative, but 50k sentences are roughly a good maximum). To use them, we simply plug in a trained (fitted) pipeline and then annotate a plain text. We don't even need to convert the input text to DataFrame in order to feed it into a pipeline that's accepting DataFrame as an input in the first place. This feature would be quite useful when it comes to getting a prediction for a few lines of text from a trained ML model.

# COMMAND ----------

light_model = LightPipeline(model)

# COMMAND ----------

lp_results = light_model.fullAnnotate(sample_text)

# COMMAND ----------

# MAGIC %md
# MAGIC **RE Results with LP Function**
# MAGIC 
# MAGIC Lets create function to get the results in a pandas dataframe.

# COMMAND ----------

def get_relations_df (results, col='relations'):
  rel_pairs=[]
  for rel in results[0][col]:
      rel_pairs.append((
          rel.metadata['sentence'],
          rel.metadata['entity1_begin'],
          rel.metadata['entity1_end'],
          rel.metadata['chunk1'], 
          rel.metadata['entity1'], 
          rel.metadata['entity2_begin'],
          rel.metadata['entity2_end'],
          rel.metadata['chunk2'], 
          rel.metadata['entity2'],
          rel.metadata['hypothesis'],
          rel.metadata['nli_prediction'],
          rel.result, 
          rel.metadata['confidence'],
      ))

  rel_df = pd.DataFrame(rel_pairs, columns=['sentence', 'entity1_begin','entity1_end','chunk1','entity1','entity2_begin','entity2_end','chunk2', 'entity2','hypothesis', 'nli_prediction', 'relation', 'confidence'])

  return rel_df

# COMMAND ----------

rel_df = get_relations_df(lp_results)

print(sample_text, "\n\n")
rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC **As you can see, the results of LP are the same with Spark DF.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization of Extracted Relations
# MAGIC 
# MAGIC We use `RelationExtractionVisualizer` method of `spark-nlp-display` library for visualization fo the extracted relations between the entities.

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer


vis = RelationExtractionVisualizer()
re_vis = vis.display(lp_results[0], 'relations', show_relations=True, return_html=True)
displayHTML(re_vis)

# COMMAND ----------

# another example

sample_text2 = "After taking Lipitor, I experienced fatigue and anxiety."

lp_results2 = light_model.fullAnnotate(sample_text2)
re_vis = vis.display(lp_results2[0], 'relations', show_relations=True, return_html=True)
displayHTML(re_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the Model and Load from Disc
# MAGIC 
# MAGIC After creating the Hypothesis, we can save the model and load it from disc without setting releatin categories.

# COMMAND ----------

# save model

re_model.write().overwrite().save("dbfs:/databricks/driver/zeroshot_re_model")

# COMMAND ----------

# load from disc and create a new pipeline

re_model2 = ZeroShotRelationExtractionModel.load("dbfs:/databricks/driver/zeroshot_re_model")\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")


pipeline2 = sparknlp.base.Pipeline() \
    .setStages([documenter,  
                sentencer,
                tokenizer, 
                words_embedder, 
                ner_clinical, 
                ner_clinical_converter,
                ner_posology, 
                ner_posology_converter,
                chunk_merger,
                pos_tagger, 
                dependency_parser, 
                re_ner_chunk_filter, 
                re_model2])
    
model2 = pipeline2.fit(data)
results2 = model2.transform(data)

# COMMAND ----------

# results of the new pipeline

results2.selectExpr("explode(relations) as relation").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see above, we got the same results by loading our saved ZSL model from disc although we didn't set any relation categories.