# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Chunk Mapping

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
# MAGIC # 1- Pretrained Chunk Mapper Models and Pretrained Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC **<center>MAPPER MODELS**
# MAGIC 
# MAGIC |index|model|index|model|index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|-----:|:-----|-----:|:-----|
# MAGIC | 1| [abbreviation_mapper](https://nlp.johnsnowlabs.com/2022/05/11/abbreviation_mapper_en_3_0.html)  | 8| [icd9_icd10_mapper](https://nlp.johnsnowlabs.com/2022/09/30/icd9_icd10_mapper_en.html)  | 15| [rxnorm_ndc_mapper](https://nlp.johnsnowlabs.com/2022/05/20/rxnorm_ndc_mapper_en_3_0.html)  | 22| [umls_clinical_findings_mapper](https://nlp.johnsnowlabs.com/2022/07/08/umls_clinical_findings_mapper_en_3_0.html)  |
# MAGIC | 2| [drug_action_treatment_mapper](https://nlp.johnsnowlabs.com/2022/03/31/drug_action_treatment_mapper_en_3_0.html)  | 9| [icd9_mapper](https://nlp.johnsnowlabs.com/2022/09/30/icd9_mapper_en.html)  | 16| [rxnorm_normalized_mapper](https://nlp.johnsnowlabs.com/2022/09/29/rxnorm_normalized_mapper_en.html)  | 23| [umls_disease_syndrome_mapper](https://nlp.johnsnowlabs.com/2022/07/11/umls_disease_syndrome_mapper_en_3_0.html)  |
# MAGIC | 3| [drug_ade_mapper](https://nlp.johnsnowlabs.com/2022/08/23/drug_ade_mapper_en.html)  | 10| [icdo_snomed_mapper](https://nlp.johnsnowlabs.com/2022/06/26/icdo_snomed_mapper_en_3_0.html)  | 17| [rxnorm_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/26/rxnorm_umls_mapper_en_3_0.html)  | 24| [umls_drug_substance_mapper](https://nlp.johnsnowlabs.com/2022/07/11/umls_drug_substance_mapper_en_3_0.html)  |
# MAGIC | 4| [drug_brandname_ndc_mapper](https://nlp.johnsnowlabs.com/2022/05/11/drug_brandname_ndc_mapper_en_3_0.html)  | 11| [mesh_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/26/mesh_umls_mapper_en_3_0.html)  | 18| [snomed_icd10cm_mapper](https://nlp.johnsnowlabs.com/2022/06/26/snomed_icd10cm_mapper_en_3_0.html)  | 25| [umls_major_concepts_mapper](https://nlp.johnsnowlabs.com/2022/07/11/umls_major_concepts_mapper_en_3_0.html)  |
# MAGIC | 5| [icd10_icd9_mapper](https://nlp.johnsnowlabs.com/2022/09/30/icd10_icd9_mapper_en.html)  | 12| [normalized_section_header_mapper](https://nlp.johnsnowlabs.com/2022/06/26/normalized_section_header_mapper_en_3_0.html)  | 19| [snomed_icdo_mapper](https://nlp.johnsnowlabs.com/2022/06/26/snomed_icdo_mapper_en_3_0.html)  |
# MAGIC | 6| [icd10cm_snomed_mapper](https://nlp.johnsnowlabs.com/2022/06/26/icd10cm_snomed_mapper_en_3_0.html)  | 13| [rxnorm_action_treatment_mapper](https://nlp.johnsnowlabs.com/2022/05/08/rxnorm_action_treatment_mapper_en_3_0.html)  | 20| [snomed_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/27/snomed_umls_mapper_en_3_0.html)  |
# MAGIC | 7| [icd10cm_umls_mapper](https://nlp.johnsnowlabs.com/2022/06/26/icd10cm_umls_mapper_en_3_0.html)  | 14| [rxnorm_mapper](https://nlp.johnsnowlabs.com/2022/06/27/rxnorm_mapper_en_3_0.html)  | 21| [umls_clinical_drugs_mapper](https://nlp.johnsnowlabs.com/2022/07/06/umls_clinical_drugs_mapper_en_3_0.html)  | 
# MAGIC 
# MAGIC 
# MAGIC **You can find all these models and more [NLP Models Hub](https://nlp.johnsnowlabs.com/models?q=Chunk+Mapping&edition=Spark+NLP+for+Healthcare)**
# MAGIC 
# MAGIC <br>

# COMMAND ----------

# MAGIC %md
# MAGIC **<center>PRETRAINED MAPPER PIPELINES**
# MAGIC 
# MAGIC 
# MAGIC | Pipeline Name          | Source    | Target    |
# MAGIC |------------------------|-----------|-----------|
# MAGIC | snomed_icd10cm_mapping | SNOMED CT | ICD-10-CM |
# MAGIC | icdo_snomed_mapping    | ICD-O     | SNOMED CT |
# MAGIC | snomed_icdo_mapping    | SNOMED CT | ICD-O     |
# MAGIC | rxnorm_ndc_mapping     | RxNorm    | NDC       |
# MAGIC | icd10cm_umls_mapping   | ICD-10-CM | UMLS      |
# MAGIC | mesh_umls_mapping      | MeSH      | UMLS      |
# MAGIC | rxnorm_umls_mapping    | RxNorm    | UMLS      |
# MAGIC | snomed_umls_mapping    | SNOMED CT | UMLS      |
# MAGIC | icd10_icd9_mapping     | ICD-10-CM | ICD-9     |
# MAGIC | rxnorm_mesh_mapping     | RxNorm | MeSH     |
# MAGIC | icd10cm_snomed_mapping     | ICD-10-CM | SNOMED CT     |
# MAGIC   
# MAGIC You can check [Healthcare Code Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb) for the examples of pretrained mapper pipelines.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1- Drug Action Treatment Mapper

# COMMAND ----------

# MAGIC %md
# MAGIC Pretrained `drug_action_treatment_mapper` model maps drugs with their corresponding `action` and `treatment` through `ChunkMapperModel()` annotator. <br/>
# MAGIC 
# MAGIC 
# MAGIC **Action** of drug refers to the function of a drug in various body systems. <br/>
# MAGIC **Treatment** refers to which disease the drug is used to treat. 
# MAGIC 
# MAGIC We can choose which option we want to use by setting `setRels()` parameter of `ChunkMapperModel()`

# COMMAND ----------

# MAGIC %md
# MAGIC We will create a pipeline consisting `bert_token_classifier_drug_development_trials` ner model to extract ner chunk as well as `ChunkMapperModel()`. <br/>
# MAGIC  Also, we will set the `.setRels()` parameter with `action` and see the results.

# COMMAND ----------

#ChunkMapper Pipeline
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

ner =  MedicalBertForTokenClassifier.pretrained("bert_token_classifier_drug_development_trials", "en", "clinical/models")\
      .setInputCols("token","sentence")\
      .setOutputCol("ner")

nerconverter = NerConverter()\
      .setInputCols("sentence", "token", "ner")\
      .setOutputCol("ner_chunk")

#drug_action_treatment_mapper with "action" mappings
chunkerMapper= ChunkMapperModel().pretrained("drug_action_treatment_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("action_mappings")\
    .setRels(["action"])
    

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer,
                                 ner, 
                                 nerconverter, 
                                 chunkerMapper])

text = [["""The patient was female and patient of Dr. X. and she was given Dermovate, Aspagin"""]]


test_data = spark.createDataFrame(text).toDF("text")

res = pipeline.fit(test_data).transform(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Chunks detected by ner model

# COMMAND ----------

res.select(F.explode('ner_chunk.result').alias("chunks")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking mapping results

# COMMAND ----------

res.select("action_mappings.result").show(truncate=False)

# COMMAND ----------

res.selectExpr("action_mappings.metadata").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see above under the ***metadata*** column, if exist, we can see all the relations for each chunk. <br/>

# COMMAND ----------

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.action_mappings.result, res.action_mappings.metadata)).alias("col"))\
    .select(F.expr("col['0']").alias("ner_chunk"),
            F.expr("col['1']").alias("mapping_result"),
            F.expr("col['2']['all_relations']").alias("all_relations")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's set the `.setRels(["treatment"])` and see the results.

# COMMAND ----------

#drug_action_treatment_mapper with "treatment" mappings
chunkerMapper= ChunkMapperModel().pretrained("drug_action_treatment_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("action_mappings")\
    .setRels(["treatment"])

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer,
                                 ner, 
                                 nerconverter, 
                                 chunkerMapper])

text = [
    ["""The patient was female and patient of Dr. X. and she was given Dermovate, Aspagin"""]
]

test_data = spark.createDataFrame(text).toDF("text")

res = pipeline.fit(test_data).transform(test_data)


# COMMAND ----------

res.select(F.explode('ner_chunk.result').alias("chunks")).show(truncate=False)

# COMMAND ----------

res.selectExpr("action_mappings.metadata").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Here are the ***treatment*** mappings and all relations under the metadata column.

# COMMAND ----------

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.action_mappings.result, res.action_mappings.metadata)).alias("col"))\
    .select(F.expr("col['0']").alias("ner_chunk"),
            F.expr("col['1']").alias("mapping_result"),
            F.expr("col['2']['all_relations']").alias("all_relations")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2- Section Header Normalizer Mapper

# COMMAND ----------

# MAGIC %md
# MAGIC We have `normalized_section_header_mapper` model that normalizes the section headers in clinical notes. It returns two levels of normalization called `level_1` and `level_2`. <br/>
# MAGIC 
# MAGIC **level_1** refers to the most comprehensive "section header" for the corresponding chunk while **level_2** refers to the second comprehensive one.
# MAGIC 
# MAGIC Let's create a piepline with `normalized_section_header_mapper` and see how it works

# COMMAND ----------

document_assembler = DocumentAssembler()\
       .setInputCol('text')\
       .setOutputCol('document')

sentence_detector = SentenceDetector()\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en","clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_jsl_slim", "en", "clinical/models")\
      .setInputCols(["sentence","token", "word_embeddings"])\
      .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Header"])

chunkerMapper = ChunkMapperModel.pretrained("normalized_section_header_mapper", "en", "clinical/models") \
       .setInputCols("ner_chunk")\
       .setOutputCol("mappings")\
       .setRels(["level_1"]) #or level_2

pipeline = Pipeline().setStages([document_assembler,
                                sentence_detector,
                                tokenizer, 
                                embeddings,
                                clinical_ner, 
                                ner_converter, 
                                chunkerMapper])

sentences = [
    ["""ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelioma.
        PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma.
        GENERAL REVIEW Right pleural effusion, firm nodules, diffuse scattered throughout the right pleura and diaphragmatic surface.
    """]]

test_data = spark.createDataFrame(sentences).toDF("text")
res = pipeline.fit(test_data).transform(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the headers detected by ner model

# COMMAND ----------

res.select(F.explode('ner_chunk.result').alias("chunks")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking mapping results

# COMMAND ----------

res.select("mappings.result").show(truncate=False)

# COMMAND ----------

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.mappings.result)).alias("col"))\
    .select(F.expr("col['0']").alias("ner_chunk"),
            F.expr("col['1']").alias("mapping_result")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see above, we can see the "level_1" based normalized version of each section header.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3- Drug Brand Name NDC Mapper

# COMMAND ----------

# MAGIC %md
# MAGIC We have `drug_brandname_ndc_mapper` model that maps drug brand names to corresponding National Drug Codes (NDC). Product NDCs for each strength are returned in result and metadata. <br/>
# MAGIC 
# MAGIC It has one relation type called `Strength_NDC`

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a pipeline with `drug_brandname_ndc_mapper` and see how it works.

# COMMAND ----------

document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("chunk")

chunkerMapper = ChunkMapperModel.pretrained("drug_brandname_ndc_mapper", "en", "clinical/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("ndc")\
      .setRels(["Strength_NDC"])

pipeline = Pipeline().setStages([document_assembler,
                                 chunkerMapper])  

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 

lp = LightPipeline(model)

res = lp.fullAnnotate('ZYVOX')

# COMMAND ----------

chunks = []
mappings = []
all_re= []

for m, n in list(zip(res[0]['chunk'], res[0]["ndc"])):
        
    chunks.append(m.result)
    mappings.append(n.result) 
    all_re.append(n.metadata["all_relations"])
    
import pandas as pd
pd.set_option('display.max_colwidth', None)

df = pd.DataFrame({'Brand_Name':chunks, 'Strenth_NDC': mappings, 'Other_NDC':all_re})

df

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, we can see corresponding "NDC" mappings of each "brand names".

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4- RxNorm NDC Mapper

# COMMAND ----------

# MAGIC %md
# MAGIC We have `rxnorm_ndc_mapper` model that maps RxNorm and RxNorm Extension codes with corresponding National Drug Codes (NDC).
# MAGIC 
# MAGIC It has two relation types that can be defined in `setRel()` parameter; **Product NDC** and **Package NDC**

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a pipeline with `rxnorm_ndc_mapper` model by setting the  relation as `setRel("Product NDC")` and see the results.

# COMMAND ----------

document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('ner_chunk')

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sentence_embeddings")\
      .setCaseSensitive(False)
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

chunkerMapper_product = ChunkMapperModel.pretrained("rxnorm_ndc_mapper", "en", "clinical/models")\
      .setInputCols(["rxnorm_code"])\
      .setOutputCol("Product NDC")\
      .setRels(["Product NDC"]) #or Package NDC

pipeline = Pipeline().setStages([document_assembler,
                                 sbert_embedder,
                                 rxnorm_resolver,
                                 chunkerMapper_product
                                 ])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 

lp = LightPipeline(model)

result = lp.fullAnnotate('macadamia nut 100 MG/ML')

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the results

# COMMAND ----------

chunks = []
rxnorm_code = []
product= []


for m, n, j in list(zip(result[0]['ner_chunk'], result[0]["rxnorm_code"], result[0]["Product NDC"])):

    chunks.append(m.result)
    rxnorm_code.append(n.result) 
    product.append(j.result)
    
import pandas as pd

df = pd.DataFrame({'ner_chunk':chunks,
                   'rxnorm_code': rxnorm_code,
                   'Product NDC': product})

df

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, we can see corresponding "Product NDC" mappings of each "RxNorm codes".

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5- RxNorm Action Treatment Mapper

# COMMAND ----------

# MAGIC %md
# MAGIC We have `rxnorm_action_treatment_mapper` model that maps RxNorm and RxNorm Extension codes with their corresponding action and treatment. It has two relation types that can be defined in `setRel()` parameter; <br/>
# MAGIC 
# MAGIC **Action** of drug refers to the function of a drug in various body systems. <br/>
# MAGIC **Treatment** refers to which disease the drug is used to treat.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a pipeline and see how it works.

# COMMAND ----------

document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('ner_chunk')

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sentence_embeddings")\
      .setCaseSensitive(False)
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

chunkerMapper_action = ChunkMapperModel.pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")\
      .setInputCols(["rxnorm_code"])\
      .setOutputCol("Action")\
      .setRels(["action"]) #or treatment

pipeline = Pipeline().setStages([document_assembler,
                                 sbert_embedder,
                                 rxnorm_resolver,
                                 chunkerMapper_action
                                 ])

model = pipeline.fit(spark.createDataFrame([['']]).toDF('text')) 

lp = LightPipeline(model)

res = lp.fullAnnotate('Zonalon 50 mg')

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the results

# COMMAND ----------

chunks = []
rxnorm_code = []
action= []


for m, n, j in list(zip(res[0]['ner_chunk'], res[0]["rxnorm_code"], res[0]["Action"])):

    chunks.append(m.result)
    rxnorm_code.append(n.result) 
    action.append(j.result)
    
import pandas as pd

df = pd.DataFrame({'ner_chunk':chunks,
                   'rxnorm_code': rxnorm_code,
                   'Action': action})

df

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, we can see corresponding "Action" mappings of each "RxNorm codes".

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6- Abbreviation Mapper

# COMMAND ----------

# MAGIC %md
# MAGIC We have `abbreviation_mapper` model that maps abbreviations and acronyms of medical regulatory activities with their definitions. <br/> It has one relation type that can be defined in `setRels(["definition"])` parameter.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a pipeline consisting `ner_abbreviation_clinical` to extract abbreviations from text, and feed the `abbreviation_mapper` with it.

# COMMAND ----------

document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

#NER model to detect abbreviations in the text
abbr_ner = MedicalNerModel.pretrained('ner_abbreviation_clinical', 'en', 'clinical/models') \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("abbr_ner")

abbr_converter = NerConverter() \
      .setInputCols(["sentence", "token", "abbr_ner"]) \
      .setOutputCol("abbr_ner_chunk")\

chunkerMapper = ChunkMapperModel.pretrained("abbreviation_mapper", "en", "clinical/models")\
      .setInputCols(["abbr_ner_chunk"])\
      .setOutputCol("mappings")\
      .setRels(["definition"]) 

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 abbr_ner, 
                                 abbr_converter, 
                                 chunkerMapper])

text = ["""Gravid with estimated fetal weight of 6-6/12 pounds.
           LABORATORY DATA: Laboratory tests include a CBC which is normal. 
           HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the results

# COMMAND ----------

#abbreviations extracted by ner model
res.select("abbr_ner_chunk.result").show()

# COMMAND ----------

res.select(F.explode(F.arrays_zip(res.abbr_ner_chunk.result, res.mappings.result)).alias("col"))\
    .select(F.expr("col['0']").alias("Abbreviation"),
            F.expr("col['1']").alias("Definition")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, we can see corresponding "definition" mappings of each "abbreviation".

# COMMAND ----------

# MAGIC %md
# MAGIC # 2- Creating a Mapper Model

# COMMAND ----------

# MAGIC %md
# MAGIC There is a `ChunkMapperApproach()` to create your own mapper model. <br/>
# MAGIC 
# MAGIC This receives an `ner_chunk` and a Json with a mapping of ner entities and relations, and returns the `ner_chunk` augmented with the relations from the Json ontology. <br/> We give the path of json file to the `setDictionary()` parameter.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create an example Json, then create a drug mapper model. This model will match the given drug name (only "metformin" for our example) with correpsonding action and treatment.  
# MAGIC 
# MAGIC The format of json file should be like following:

# COMMAND ----------

data_set= {
  "mappings": [
    {
      "key": "metformin",
      "relations": [
        {
          "key": "action",
          "values" : ["hypoglycemic", "Drugs Used In Diabetes"]
        },
        {
          "key": "treatment",
          "values" : ["diabetes", "t2dm"]
        }
      ]
    }
  ]
}

import json
with open('/dbfs/sample_drug.json', 'w', encoding='utf-8') as f:
    json.dump(data_set, f, ensure_ascii=False, indent=4)

# COMMAND ----------

# MAGIC %md
# MAGIC By using `setRel()` parameter, we tell the model which type of mapping we want. In our case, if we want from our model to return **action** mapping, we set the parameter as `setRels(["action"])`,  we set as `setRels(["treatment"])` for **treatment**
# MAGIC 
# MAGIC Let's create a pipeline and see it in action.

# COMMAND ----------

document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

#NER model to detect drug in the text
clinical_ner = MedicalNerModel.pretrained("ner_posology_small","en","clinical/models")\
	    .setInputCols(["sentence","token","embeddings"])\
	    .setOutputCol("ner")\
      .setLabelCasing("upper")
 
ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["DRUG"])

chunkerMapper = ChunkMapperApproach()\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setDictionary("file:/dbfs/sample_drug.json")\
      .setRels(["action"]) #or treatment

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])

text = ["The patient was given 1 unit of metformin daily."]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)


# COMMAND ----------

res.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the ner result

# COMMAND ----------

res.select(F.explode('ner_chunk.result').alias("chunks")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking the mapper result

# COMMAND ----------

res.selectExpr("mappings.metadata").show(truncate=False)

# COMMAND ----------

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.mappings.result, res.mappings.metadata)).alias("col"))\
    .select(F.expr("col['0']").alias("ner_chunk"),
            F.expr("col['1']").alias("mapping_result"),
            F.expr("col['2']['all_relations']").alias("all_relations")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, the model that we created with `ChunkMapperApproach()` succesfully mapped "metformin". Under the metadata, we can see all relations that we defined in the Json.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1- Save the model to disk

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we will save our model and use it with `ChunkMapperModel()`

# COMMAND ----------

model.stages[-1].write().save("dbfs:/databricks/driver/models/drug_mapper")

# COMMAND ----------

# MAGIC %md
# MAGIC Using the saved model. This time we will check 'treatment' mappings results

# COMMAND ----------

document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

#NER model to detect drug in the text
clinical_ner = MedicalNerModel.pretrained("ner_posology_small","en","clinical/models")\
	    .setInputCols(["sentence","token","embeddings"])\
	    .setOutputCol("ner")\
      .setLabelCasing("upper")
 
ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")\
      .setWhiteList(["DRUG"])

chunkerMapper = ChunkMapperModel.load("dbfs:/databricks/driver/models/drug_mapper")\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("mappings")\
      .setRels(["treatment"]) 

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])

text = ["The patient was given 1 unit of metformin daily."]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)

# COMMAND ----------

res.selectExpr("mappings.metadata").show(truncate=False)

# COMMAND ----------

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.mappings.result, res.mappings.metadata)).alias("col"))\
    .select(F.expr("col['0']").alias("ner_chunk"),
            F.expr("col['1']").alias("mapping_result"),
            F.expr("col['2']['all_relations']").alias("all_relations")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see above, we created our own drug mapper model successfully.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2- Create a Model with Upper Cased or Lower Cased

# COMMAND ----------

# MAGIC %md
# MAGIC We can set the case status of `ChunkMapperApproach` while creating a model by using `setLowerCase()` parameter.
# MAGIC 
# MAGIC Let's create a new mapping dictionary and see how it works.

# COMMAND ----------

data_set= {
    "mappings": [
        {
            "key": "Warfarina lusa",
            "relations": [
                {
                    "key": "action",
                    "values": [
                        "Analgesic",
                        "Antipyretic"
                    ]
                },
                {
                    "key": "treatment",
                    "values": [
                        "diabetes",
                        "t2dm"
                    ]
                }
            ]
        }
    ]
}

import json
with open('/dbfs/mappings.json', 'w', encoding='utf-8') as f:
    json.dump(data_set, f, ensure_ascii=False, indent=4)

# COMMAND ----------

sentences = [
        ["""The patient was given Warfarina Lusa and amlodipine 10 MG.The patient was given Aspagin, coumadin 5 mg, coumadin, and he has metamorfin"""]
    ]


test_data = spark.createDataFrame(sentences).toDF("text")

# COMMAND ----------

# MAGIC %md
# MAGIC **`setLowerCase(True)`**

# COMMAND ----------

chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("file:/dbfs/mappings.json") \
        .setRels(["action"]) \
        .setLowerCase(True) \

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])


result_df = pipeline.fit(test_data).transform(test_data)
result_df.selectExpr("explode(mappings)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC "Warfarina lusa" is in lower case in the source json file, and in upper case(Warfarina Lusa) in our example training sentence. We trained that model in lower case, the model mapped the entity even though our training sentence is uppercased. <br/>
# MAGIC 
# MAGIC Let's check with `setLowerCase(False)` and see the difference.

# COMMAND ----------

chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("file:/dbfs/mappings.json") \
        .setRels(["action"]) \
        .setLowerCase(False) \

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])


result_df = pipeline.fit(test_data).transform(test_data)
result_df.selectExpr("explode(mappings)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, our model couldn't map the given uppercased "Warfarine Lura".

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3- Selecting Multiple Relations

# COMMAND ----------

# MAGIC %md
# MAGIC We can select multiple relations for the same chunk with the `setRels()` parameter.

# COMMAND ----------

chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("file:/dbfs/mappings.json") \
        .setLowerCase(True) \
        .setRels(["action", "treatment"])

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])


result_df = pipeline.fit(test_data).transform(test_data)
result_df.selectExpr("explode(mappings)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, we are able to see all the relations(action, treatment) at the same time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4- Filtering Multi-token Chunks

# COMMAND ----------

# MAGIC %md
# MAGIC If the chunk includes multi-tokens splitted by a whitespace, we can filter that chunk by using `setAllowMultiTokenChunk()` parameter.

# COMMAND ----------

chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("file:/dbfs/mappings.json") \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \
        .setAllowMultiTokenChunk(False)

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])


result_df = pipeline.fit(test_data).transform(test_data)
result_df.selectExpr("explode(mappings)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC The chunk "Warfarina Lusa" is a multi-token. Therefore, our mapper model skip that entity. <br/>
# MAGIC So, let's set `.setAllowMultiTokenChunk(True)` and see the difference.

# COMMAND ----------

chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("file:/dbfs/mappings.json") \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \
        .setAllowMultiTokenChunk(True)

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])


result_df = pipeline.fit(test_data).transform(test_data)
result_df.selectExpr("explode(mappings)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3- ChunkMapperFilterer

# COMMAND ----------

# MAGIC %md
# MAGIC `ChunkMapperFilterer` annotator allows filtering of the chunks that were passed through the ChunkMapperModel. <br/>
# MAGIC 
# MAGIC We can filter chunks by setting the `.setReturnCriteria()` parameter. It has 2 options; <br/>
# MAGIC 
# MAGIC 
# MAGIC **success:** Returns the chunks which are mapped by ChunkMapper <br/>
# MAGIC 
# MAGIC **fail:** Returns the chunks which are not mapped by ChunkMapper <br/>

# COMMAND ----------

# MAGIC %md
# MAGIC Let's apply the both options and check the results.

# COMMAND ----------

chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("file:/dbfs/mappings.json") \
        .setRel("action") \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 word_embeddings,
                                 clinical_ner, 
                                 ner_converter, 
                                 chunkerMapper])


result_df = pipeline.fit(test_data).transform(test_data)
result_df.selectExpr("explode(mappings)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **`.setReturnCriteria("success")`**

# COMMAND ----------

cfModel = ChunkMapperFilterer() \
        .setInputCols(["ner_chunk","mappings"]) \
        .setOutputCol("chunks_filtered")\
        .setReturnCriteria("success")

cfModel.transform(result_df).selectExpr("explode(chunks_filtered)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **`.setReturnCriteria("fail")`**

# COMMAND ----------

cfModel = ChunkMapperFilterer() \
        .setInputCols(["ner_chunk","mappings"]) \
        .setOutputCol("chunks_filtered")\
        .setReturnCriteria("fail")

cfModel.transform(result_df).selectExpr("explode(chunks_filtered)").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4- ResolverMerger - Using Sentence Entity Resolver and `ChunkMapperModel` Together

# COMMAND ----------

# MAGIC %md
# MAGIC We can merge the results of `ChunkMapperModel` and `SentenceEntityResolverModel` by using `ResolverMerger` annotator. 
# MAGIC 
# MAGIC We can detect our results that fail by `ChunkMapperModel` with `ChunkMapperFilterer` and then merge the resolver and mapper results with `ResolverMerger`

# COMMAND ----------

document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")\
      .setInputCols(["sentence", "token", "embeddings"])\
      .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols("sentence", "token", "ner")\
      .setOutputCol("chunk")

chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("RxNorm_Mapper")\
      .setRels(["rxnorm_code"])

cfModel = ChunkMapperFilterer() \
      .setInputCols(["chunk", "RxNorm_Mapper"]) \
      .setOutputCol("chunks_fail") \
      .setReturnCriteria("fail")

chunk2doc = Chunk2Doc() \
      .setInputCols("chunks_fail") \
      .setOutputCol("chunk_doc")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["chunk_doc"])\
      .setOutputCol("sentence_embeddings")\
      .setCaseSensitive(False)

resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented", "en", "clinical/models") \
      .setInputCols(["chunks_fail", "sentence_embeddings"]) \
      .setOutputCol("resolver_code") \
      .setDistanceFunction("EUCLIDEAN")

resolverMerger = ResolverMerger()\
      .setInputCols(["resolver_code","RxNorm_Mapper"])\
      .setOutputCol("RxNorm")

mapper_pipeline = Pipeline(
      stages = [
          document_assembler,
          sentence_detector,
          tokenizer,
          word_embeddings,
          ner_model,
          ner_converter,
          chunkerMapper,
          chunkerMapper,
          cfModel,
          chunk2doc,
          sbert_embedder,
          resolver,
          resolverMerger
      ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = mapper_pipeline.fit(empty_data)

# COMMAND ----------

samples = [['The patient was given Adapin 10 MG, coumadn 5 mg'],
           ['The patient was given Avandia 4 mg, Tegretol, zitiga'] ]

result = model.transform(spark.createDataFrame(samples).toDF("text"))

# COMMAND ----------

result.selectExpr('chunk.result as chunk', 
                  'RxNorm_Mapper.result as RxNorm_Mapper', 
                  'chunks_fail.result as chunks_fail', 
                  'resolver_code.result as resolver_code',
                  'RxNorm.result as RxNorm'
              ).show(truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5- Section Header Normalizer Mapper with ChunkSentenceSplitter

# COMMAND ----------

# MAGIC %md
# MAGIC `ChunkSentenceSplitter()` annotator splits documents or sentences by chunks provided. <br/> For detailed usage of this annotator, visit [this notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/18.Chunk_Sentence_Splitter.ipynb) <br/>
# MAGIC 
# MAGIC In this section, we will do the following steps; 
# MAGIC - Detect "section headers" in given text through Ner model
# MAGIC - Split the given text by headers with `ChunkSentenceSplitter()`
# MAGIC - Normalize the `ChunkSentenceSplitter()` outputs with `normalized_section_header_mapper` model.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start with creating Ner pipeline to detect "Header"

# COMMAND ----------

sentences = [
    ["""ADMISSION DIAGNOSIS Right pleural effusion and suspected malignant mesothelioma.
        PRINCIPAL DIAGNOSIS Right pleural effusion, suspected malignant mesothelioma.
        REVIEW OF SYSTEMS Right pleural effusion, firm nodules, diffuse scattered throughout the right pleura and diaphragmatic surface.
    """]]

df= spark.createDataFrame(sentences).toDF("text")

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

tokenizer= Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_jsl_slim", "en", "clinical/models")\
      .setInputCols("token", "document")\
      .setOutputCol("ner")\
      .setCaseSensitive(True)

ner_converter = NerConverter() \
      .setInputCols(["document", "token", "ner"]) \
      .setOutputCol("ner_chunk")\
      .setWhiteList(["Header"])

pipeline = Pipeline(
    stages = [
        documentAssembler,
        tokenizer,
        tokenClassifier,
        ner_converter
    ])
 
empty_df = spark.createDataFrame([[""]]).toDF('text')
pipeline_model = pipeline.fit(empty_df)

# COMMAND ----------

result = pipeline_model.transform(df)
result.selectExpr('explode(ner_chunk)').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have our header entities. We will split the text by the headers.

# COMMAND ----------

#applying ChunkSentenceSplitter 
chunkSentenceSplitter = ChunkSentenceSplitter()\
    .setInputCols("document","ner_chunk")\
    .setOutputCol("paragraphs")\
    .setGroupBySentences(False)

paragraphs = chunkSentenceSplitter.transform(result)

# COMMAND ----------

paragraphs.show()

# COMMAND ----------

pd.set_option('display.max_colwidth', None)
result_df = paragraphs.selectExpr("explode(paragraphs) as result").selectExpr("result.result","result.metadata.entity", "result.metadata.splitter_chunk").toPandas()
result_df

# COMMAND ----------

# MAGIC %md
# MAGIC As you see, we have our splitted text and **section headers**. <br/>
# MAGIC Now we will normalize this section headers with `normalized_section_header_mapper`

# COMMAND ----------

chunkerMapper = ChunkMapperModel.pretrained("normalized_section_header_mapper", "en", "clinical/models") \
       .setInputCols("ner_chunk")\
       .setOutputCol("mappings")\
       .setRels(["level_1"]) #or level_2

normalized_df= chunkerMapper.transform(paragraphs)

# COMMAND ----------

normalized_df.show()

# COMMAND ----------

normalized_df= normalized_df.select(F.explode(F.arrays_zip(normalized_df.ner_chunk.result, 
                                                           normalized_df.mappings.result)).alias("col"))\
                            .select(F.expr("col['0']").alias("ner_chunk"),
                                    F.expr("col['1']").alias("normalized_headers")).toPandas()
normalized_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have our normalized headers. We will merge it with `ChunkSentenceSplitter()` output

# COMMAND ----------

normalized_df= normalized_df.rename(columns={"ner_chunk": "splitter_chunk"})
df= pd.merge(result_df, normalized_df, on=["splitter_chunk"])

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC Ultimately, we have splitted paragraphs, headers and normalized headers.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5- Pretrained Mapper Pipelines
# MAGIC 
# MAGIC We will show an example of `rxnorm_umls_mapping` pipeline here. But you can check [Healthcare Code Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb) for the examples of pretrained mapper pipelines.

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

rxnorm_umls_pipeline= PretrainedPipeline("rxnorm_umls_mapping", "en", "clinical/models")

# COMMAND ----------

rxnorm_umls_pipeline.annotate("1161611 315677 343663")

# COMMAND ----------

# MAGIC %md
# MAGIC |**RxNorm Code** | **RxNorm Details** | **UMLS Code** | **UMLS Details** |
# MAGIC | ---------- | -----------:| ---------- | -----------:|
# MAGIC | 1161611 |  metformin Pill | C3215948 | metformin pill |
# MAGIC | 315677 | cimetidine 100 mg | C0984912 | cimetidine 100 mg |
# MAGIC | 343663 | insulin lispro 50 UNT/ML | C1146501 | insulin lispro 50 unt/ml |