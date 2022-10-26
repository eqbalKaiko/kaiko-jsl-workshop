# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Merging Annotations From Multiple Named Entity Recognition Models

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
import warnings
warnings.filterwarnings('ignore')

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)


print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

# Sample data
data_chunk_merge = spark.createDataFrame([
  (1,"""A 63 years old man presents to the hospital with a history of recurrent infections that include cellulitis, pneumonias, and upper respiratory tract infections. He reports subjective fevers at home along with unintentional weight loss and occasional night sweats. The patient has a remote history of arthritis, which was diagnosed approximately 20 years ago and treated intermittently with methotrexate (MTX) and prednisone. On physical exam, he is found to be febrile at 102°F, rather cachectic, pale, and have hepatosplenomegaly. Several swollen joints that are tender to palpation and have decreased range of motion are also present. His laboratory values show pancytopenia with the most severe deficiency in neutrophils.
""")]).toDF("id","text")

data_chunk_merge.show(truncate=150)

# COMMAND ----------

# Annotator that transforms a text column from dataframe into an Annotation ready for NLP

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
clinical_ner = MedicalNerModel.pretrained("ner_deid_large", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("clinical_ner")

clinical_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "clinical_ner"]) \
    .setOutputCol("clinical_ner_chunk")

# internal clinical NER (general terms)
jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "jsl_ner"]) \
    .setOutputCol("jsl_ner_chunk")

# merge ner_chunks by prioritizing the overlapping indices (chunks with longer lengths and highest information will be kept from each ner model)
chunk_merger = ChunkMergeApproach()\
    .setInputCols('clinical_ner_chunk', "jsl_ner_chunk")\
    .setOutputCol('merged_ner_chunk')

# merge ner_chunks regardess of overlapping indices 
# only works with 2.7 and later 
chunk_merger_NonOverlapped = ChunkMergeApproach()\
    .setInputCols('clinical_ner_chunk', "jsl_ner_chunk")\
    .setOutputCol('nonOverlapped_ner_chunk')\
    .setMergeOverlapping(False)


nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    clinical_ner_converter,
    jsl_ner,
    jsl_ner_converter,
    chunk_merger,
    chunk_merger_NonOverlapped])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)


# COMMAND ----------

merged_data = model.transform(data_chunk_merge).cache()

# COMMAND ----------

merged_data.select("jsl_ner_chunk").show(truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F

result_df = merged_data.select('id',F.explode('merged_ner_chunk').alias("cols")) \
                       .select('id',F.expr("cols.begin").alias("begin"),
                               F.expr("cols.end").alias("end"),
                               F.expr("cols.result").alias("chunk"),
                               F.expr("cols.metadata.entity").alias("entity"))

result_df.show(50, truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## NonOverlapped Chunk
# MAGIC 
# MAGIC All the entities form each ner model will be returned one by one

# COMMAND ----------

from pyspark.sql import functions as F

result_df2 = merged_data.select('id',F.explode('nonOverlapped_ner_chunk').alias("cols")) \
                        .select('id',F.expr("cols.begin").alias("begin"),
                                F.expr("cols.end").alias("end"),
                                F.expr("cols.result").alias("chunk"),
                                F.expr("cols.metadata.entity").alias("entity"))

result_df2.show(50, truncate=100)


# COMMAND ----------

# MAGIC %md
# MAGIC ## ChunkMergeApproach to admit N input cols 
# MAGIC We can feed the ChunkMergerApproach more than 2 chunks, also, we can filter out the entities that we don't want to get from the ChunkMergeApproach using `setBlackList` parameter.

# COMMAND ----------

sample_text = """A 28 year old female with a history of gestational diabetes mellitus diagnosed eight years prior to 
presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis 
three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index 
( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting.
Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . 
She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . 
She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was 
significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , 
or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , 
anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin 
( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed 
as blood samples kept hemolyzing due to significant lipemia .
The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior 
to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , 
the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , 
and lipase was 52 U/L .
 β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged 
 and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . 
 The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides 
 to 1400 mg/dL , within 24 hours .
 Twenty days ago.
 Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . 
 At birth the typical boy is growing slightly faster than the typical girl, but the velocities become equal at about 
 seven months, and then the girl grows faster until four years. 
 From then until adolescence no differences in velocity 
 can be detected. 21-02-2020 
21/04/2020
"""

# COMMAND ----------

# Defining ContextualParser for feeding ChunkMergerApproach

#defining rules
date = {
  "entity": "Parser_Date",
  "ruleScope": "sentence",
  "regex": "\\d{1,2}[\\/\\-\\:]{1}(\\d{1,2}[\\/\\-\\:]{1}){0,1}\\d{2,4}",
  "valuesDefinition":[],
  "prefix": [],
  "suffix": [],
  "contextLength": 150,
  "context": []
}


with open('/dbfs/date.json', 'w') as f:
    json.dump(date, f)


age = {
  "entity": "Parser_Age",
  "ruleScope": "sentence",
  "matchScope":"token",
  "regex" : "^[1][0-9][0-9]|[1-9][0-9]|[1-9]$",
  "prefix":["age of", "age"],
  "suffix": ["-years-old",
             "years-old",
             "-year-old",
             "-months-old",
             "-month-old",
             "-months-old",
             "-day-old",
             "-days-old",
             "month old",
             "days old",
             "year old",
             "years old", 
             "years",
             "year", 
             "months", 
             "old"
              ],
  "contextLength": 25,
  "context": [],
  "contextException": ["ago"],
  "exceptionDistance": 10
}

with open("/dbfs/age.json", 'w') as f:
  json.dump(age, f)



# COMMAND ----------

# MAGIC %md
# MAGIC Using two ContextualParserApproach models and NER model in the same pipeline and merging by ChunkMergeApproach

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

# Contextual parser for age 
age_contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("entity_age") \
    .setJsonPath("/dbfs/age.json") \
    .setCaseSensitive(False) \
    .setPrefixAndSuffixMatch(False)\
    .setOptionalContextRules(False) 

chunks_age= ChunkConverter()\
    .setInputCols("entity_age")\
    .setOutputCol("chunk_age")

# Contextual parser for date
date_contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("entity_date") \
    .setJsonPath("/dbfs/date.json") \
    .setCaseSensitive(False) \
    .setPrefixAndSuffixMatch(False)

chunks_date = ChunkConverter().setInputCols("entity_date").setOutputCol("chunk_date")

# Clinical word embeddings 
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

# Extracting entities by ner_deid_large
ner_model = MedicalNerModel.pretrained("ner_deid_large","en","clinical/models") \
    .setInputCols("sentence","token","embeddings") \
    .setOutputCol("ner")

ner_converter= NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setWhiteList(["DATE", "AGE"])

# Chunkmerger; prioritize age_contextual_parser
parser_based_merge= ChunkMergeApproach()\
    .setInputCols(["chunk_age", "chunk_date", "ner_chunk"])\
    .setOutputCol("merged_chunks")

# Chunkmerger; prioritize ner_chunk
ner_based_merge= ChunkMergeApproach()\
    .setInputCols(["ner_chunk", "chunk_age", "chunk_date"])\
    .setOutputCol("merged_chunks_2")

# Using black list for limiting the entity types that will be extracted
limited_merge= ChunkMergeApproach()\
    .setInputCols(["ner_chunk", "chunk_age", "chunk_date"])\
    .setOutputCol("merged_chunks_black_list")\
    .setBlackList(["DATE", "Parser_Date"]) # this will block the dates. 

pipeline= Pipeline(stages=[
                           documentAssembler,
                           sentenceDetector,
                           tokenizer,
                           age_contextual_parser,
                           chunks_age,
                           date_contextual_parser,
                           chunks_date,
                           word_embeddings,
                           ner_model,
                           ner_converter,
                           parser_based_merge,
                           ner_based_merge,
                           limited_merge
])

empty_df= spark.createDataFrame([[""]]).toDF("text")
model= pipeline.fit(empty_df)


lmodel= LightPipeline(model)
lresult= lmodel.fullAnnotate(sample_text)[0]


# COMMAND ----------

lresult.keys()

# COMMAND ----------

# MAGIC %md
# MAGIC If there is an overlap among the input entity types, ChunkMergerApproach model prioritizes the leftmost input. <br/>
# MAGIC 
# MAGIC At the 'parser_based_merge', we gave the contextual parser's chunks firstly. Therefore, 'parser_based_merge' prioritized the "Parser_Age" and "Parser_Date" entities over the "AGE" and "DATE" entity types that comes from NER model. <br/>
# MAGIC 
# MAGIC At the 'ner_based_merge', we gave the Ner model's inputs firstly, thus 'ner_based_merge' prioritized the "AGE" and "DATE" entities over the "Parser_Age" and "Parser_Date".  <br/>
# MAGIC 
# MAGIC At the limited_merge, we excluded "DATE" and "Parser_Date" entity types.
# MAGIC 
# MAGIC Let's compare the results of these ChunkMergeApproach below:

# COMMAND ----------

chunk= []
parser_based_merge= []
ner_based_merge= []

for i, k in list(zip(lresult["merged_chunks"], list(lresult["merged_chunks_2"],))):
  parser_based_merge.append(i.metadata["entity"])
  ner_based_merge.append(k.metadata["entity"])
  chunk.append(i.result)

df= pd.DataFrame({"chunk": chunk,"parser_based_merged_entity": parser_based_merge, "ner_based_merged_entity": ner_based_merge})
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC `.setBlackList()` applied results:

# COMMAND ----------

chunk= []
limited_merge_entity= []

for i in list(lresult["merged_chunks_black_list"]):
  chunk.append(i.result)
  limited_merge_entity.append(i.metadata["entity"])

df= pd.DataFrame({"chunk": chunk, "limited_entity": limited_merge_entity }) 
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merging NERs with TextMatcher and RegexMatcher outputs in the same pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### TextMatcher

# COMMAND ----------

# MAGIC %md
# MAGIC Lets make a special NER for female using a dictionary related to female entity.

# COMMAND ----------

# write the target entities to txt file 

entities = ['she', 'her', 'girl', 'woman', 'women', 'womanish', 'womanlike', 'womanly', 'madam', 'madame', 'senora', 'lady', 'miss', 'girlfriend', 'wife', 'bride', 'misses', 'mrs.', 'female']
with open ('/dbfs/female_entities.txt', 'w') as f:
    for i in entities:
        f.write(i+'\n')

# COMMAND ----------

sample_text = """A 28 year old female with a history of gestational diabetes mellitus diagnosed eight years prior to 
presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis 
three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index 
( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting.
Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . 
She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . 
She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was 
significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , 
or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , 
anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin 
( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed 
as blood samples kept hemolyzing due to significant lipemia .
The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior 
to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , 
the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , 
and lipase was 52 U/L .
β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged 
and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . 
This madame was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides 
to 1400 mg/dL , within 24 hours .
Twenty days ago.
Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . 
At birth the typical boy is growing slightly faster than the typical girl, but the velocities become equal at about 
seven months, and then the girl grows faster until four years. 
From then until adolescence no differences in velocity 
can be detected. 21-02-2020 
21/04/2020
"""

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

# Clinical word embeddings 
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

# Extracting entities by ner_jsl
ner_model = MedicalNerModel.pretrained("ner_jsl","en","clinical/models") \
    .setInputCols("sentence","token","embeddings") \
    .setOutputCol("ner")

ner_converter= NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\

# Find female entities using TextMatcher
female_entity_extractor = TextMatcher() \
    .setInputCols(["document",'token'])\
    .setOutputCol("female_entities")\
    .setEntities("file:/dbfs/female_entities.txt")\
    .setCaseSensitive(False)\
    .setEntityValue('female_entity')

# Chunkmerger; prioritize female_entity
merger= ChunkMergeApproach()\
    .setInputCols(["female_entities", "ner_chunk"])\
    .setOutputCol("merged_chunks")

pipeline= Pipeline(stages=[
                           documentAssembler,
                           sentenceDetector,
                           tokenizer,
                           word_embeddings,
                           ner_model,
                           ner_converter,
                           female_entity_extractor,
                           merger
])

empty_df= spark.createDataFrame([[""]]).toDF("text")
model= pipeline.fit(empty_df)


tm_model= LightPipeline(model)
tm_result= tm_model.fullAnnotate(sample_text)[0]

# COMMAND ----------

chunk= []
ner = []
merge= []

for i, k in list(zip(tm_result["ner_chunk"], list(tm_result["merged_chunks"],))):
  merge.append(k.metadata["entity"])
  ner.append(i.metadata["entity"])
  chunk.append(i.result)

df= pd.DataFrame({"chunk": chunk, "ner_entity": ner, "merged_entity": merge})
df[(df.ner_entity=="Gender") | (df.merged_entity=="female_entity")]

# COMMAND ----------

# MAGIC %md
# MAGIC As seen above table, `Gender` NER entities with female info are replaced with `female_entity`. And chunk 'madame' is identified incorrectly as `Medical_Device`, but this false entity is corrected with `female_entity`, using TextMatcher annotator merging.

# COMMAND ----------

# MAGIC %md
# MAGIC If your lookup table is large, you can even use  [BigTextMatcher](https://nlp.johnsnowlabs.com/docs/en/annotators#bigtextmatcher).

# COMMAND ----------

# MAGIC %md
# MAGIC ### RegexMatcher

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will use [RegexMatcher](https://nlp.johnsnowlabs.com/docs/en/annotators#regexmatcher) to build a NER label. Initially we will build a file that contains one or multiple line regex rules. For use of RegexMather you may check [this NB](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb]).

# COMMAND ----------

rules = '''
\b[A-Z]+(\s+[A-Z]+)*:\b, SECTION_HEADER
'''

with open('/dbfs/regex_rules.txt', 'w') as f:
    
    f.write(rules)
    

# COMMAND ----------

# MAGIC %md
# MAGIC This regex rule finds `SECTION_HEADER` chunks of the document. There are some pre-trained models that can find `SECTION_HEADER`, but here we will use this method just to demonstrate the use of RegexMatcher.

# COMMAND ----------

sample_text = """
POSTOPERATIVE DIAGNOSIS: Cervical lymphadenopathy.
PROCEDURE:  Excisional biopsy of right cervical lymph node.
ANESTHESIA:  General endotracheal anesthesia.
Specimen:  Right cervical lymph node.
EBL: 10 cc.
COMPLICATIONS:  None.
FINDINGS: Enlarged level 2 lymph node was identified and removed and sent for pathologic examination.
FLUIDS:  Please see anesthesia report.
URINE OUTPUT:  None recorded during the case.
INDICATIONS FOR PROCEDURE:  This is a 43-year-old female with a several-year history of persistent cervical lymphadenopathy. She reports that it is painful to palpation on the right and has had multiple CT scans as well as an FNA which were all nondiagnostic. After risks and benefits of surgery were discussed with the patient, an informed consent was obtained. She was scheduled for an excisional biopsy of the right cervical lymph node.
PROCEDURE IN DETAIL:  The patient was taken to the operating room and placed in the supine position. She was anesthetized with general endotracheal anesthesia. The neck was then prepped and draped in the sterile fashion. Again, noted on palpation there was an enlarged level 2 cervical lymph node.A 3-cm horizontal incision was made over this lymph node. Dissection was carried down until the sternocleidomastoid muscle was identified. The enlarged lymph node that measured approximately 2 cm in diameter was identified and was removed and sent to Pathology for touch prep evaluation. The area was then explored for any other enlarged lymph nodes. None were identified, and hemostasis was achieved with electrocautery. A quarter-inch Penrose drain was placed in the wound.The wound was then irrigated and closed with 3-0 interrupted Vicryl sutures for a deep closure followed by a running 4-0 Prolene subcuticular suture. Mastisol and Steri-Strip were placed over the incision, and sterile bandage was applied. The patient tolerated this procedure well and was extubated without complications and transported to the recovery room in stable condition. She will return to the office tomorrow in followup to have the Penrose drain removed.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC Below is a typical pipeline, but RegexMatcher is added. RegexMatcher output chunks doesn't have an entity label, so we need to use [ChunkConverter](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/chunker/chunk_converter/index.html?highlight=chunkconverter#sparknlp_jsl.annotator.chunker.chunk_converter.ChunkConverter) to add entity labels to regex chunks. Finally NER and RegexMatcher (through ChunkConverter) outputs are merged  by ChunkMergeApproach.

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

# Clinical word embeddings 
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

# Extracting entities using ner_clinical_large pretrained model 
ner_model = MedicalNerModel.pretrained("ner_clinical_large","en","clinical/models") \
    .setInputCols("sentence","token","embeddings") \
    .setOutputCol("ner")

ner_converter= NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\

# Find all tokens that matches regex rule file
regex_matcher = RegexMatcher()\
    .setInputCols('document')\
    .setStrategy("MATCH_ALL")\
    .setOutputCol("regex_matches")\
    .setExternalRules(path='file:/dbfs/regex_rules.txt', delimiter=',')

# Add entity label to regex chunks to be able to merge with previous NER 
chunkConverter = ChunkConverter()\
    .setInputCols("regex_matches")\
    .setOutputCol("regex_chunk")

# Chunkmerger, prioritize regex
merger= ChunkMergeApproach()\
    .setInputCols(["regex_chunk", "ner_chunk"])\
    .setOutputCol("merged_chunks")\
    .setMergeOverlapping(True)\
    .setChunkPrecedence("field")

pipeline= Pipeline(stages=[
                           documentAssembler,
                           sentenceDetector,
                           tokenizer,
                           word_embeddings,
                           ner_model,
                           ner_converter,
                           regex_matcher,
                           chunkConverter,
                           merger
])

empty_df= spark.createDataFrame([[""]]).toDF("text")
model= pipeline.fit(empty_df)

rm_model= LightPipeline(model)
rm_result=rm_model.fullAnnotate(sample_text)[0]


# COMMAND ----------

rm_result["regex_chunk"]

# COMMAND ----------

chunk= []
ner = []
for i in list(rm_result["ner_chunk"]):
  ner.append(i.metadata["entity"])
  chunk.append(i.result)
df_ner = pd.DataFrame({"chunk": chunk,  "ner_entity": ner})

chunk= []
regex = []
for i in list(rm_result["regex_chunk"]):
  regex.append(i.metadata["entity"])
  chunk.append(i.result)
df_regex = pd.DataFrame({"chunk": chunk,  "ner_entity": regex})

chunk= []
merge= []
for i in list(rm_result["merged_chunks"]):
  merge.append(i.metadata["entity"])
  chunk.append(i.result)
df_merge = pd.DataFrame({"chunk": chunk,  "merged_entity": merge})




# COMMAND ----------

# MAGIC %md
# MAGIC As seen below, `SECTION_HEADER` labels are added to merged NER listing.

# COMMAND ----------

df_ner

# COMMAND ----------

df_regex

# COMMAND ----------

df_merge