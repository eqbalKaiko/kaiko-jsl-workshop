# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Detect clinical entities, relations and assertion status with pretrained pipelines

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

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Listing Models, Pipelines and Annotators

# COMMAND ----------

# MAGIC %md
# MAGIC **You can print the list of clinical pretrained models/pipelines and annotators in Spark NLP with one-line code:**

# COMMAND ----------

from sparknlp_jsl.pretrained import InternalResourceDownloader

# print PretrainedPipelines
InternalResourceDownloader.showPrivatePipelines(lang='en')

# print models 
#InternalResourceDownloader.showPrivateModels(annotator="MedicalNerModel", lang='en')

# print annotators
# InternalResourceDownloader.showAvailableAnnotators()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pretrained Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC In order to save you from creating a pipeline from scratch, Spark NLP also has a pre-trained pipelines that are already fitted using certain annotators and transformers according to various use cases.
# MAGIC 
# MAGIC Here is the list of clinical pre-trained pipelines: 
# MAGIC 
# MAGIC > These clinical pipelines are trained with `embeddings_healthcare_100d` and accuracies might be 1-2% lower than `embeddings_clinical` which is 200d.
# MAGIC 
# MAGIC **1.   explain_clinical_doc_carp** :
# MAGIC 
# MAGIC > a pipeline with `ner_clinical`, `assertion_dl`, `re_clinical` and `ner_posology`. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.
# MAGIC 
# MAGIC **2.   explain_clinical_doc_era** :
# MAGIC 
# MAGIC > a pipeline with `ner_clinical_events`, `assertion_dl` and `re_temporal_events_clinical`. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities.
# MAGIC 
# MAGIC **3.   explain_clinical_doc_ade** :
# MAGIC 
# MAGIC > a pipeline for `Adverse Drug Events (ADE)` with `ner_ade_biobert`, `assertiondl_biobert`, `classifierdl_ade_conversational_biobert` and `re_ade_biobert`. It will classify the document, extract `ADE` and `DRUG` entities, assign assertion status to `ADE` entities, and relate them with `DRUG` entities, then assign ADE status to a text (`True` means ADE, `False` means not related to ADE).
# MAGIC 
# MAGIC **letter codes in the naming conventions:**
# MAGIC 
# MAGIC > c : ner_clinical
# MAGIC 
# MAGIC > e : ner_clinical_events
# MAGIC 
# MAGIC > r : relation extraction
# MAGIC 
# MAGIC > p : ner_posology
# MAGIC 
# MAGIC > a : assertion
# MAGIC 
# MAGIC > ade : adverse drug events
# MAGIC 
# MAGIC **Relation Extraction types:**
# MAGIC 
# MAGIC `re_clinical` >> TrIP (improved), TrWP (worsened), TrCP (caused problem), TrAP (administered), TrNAP (avoided), TeRP (revealed problem), TeCP (investigate problem), PIP (problems related)
# MAGIC 
# MAGIC `re_temporal_events_clinical` >> `AFTER`, `BEFORE`, `OVERLAP`
# MAGIC 
# MAGIC **4. Clinical Deidentification** :
# MAGIC 
# MAGIC >This pipeline can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`, `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR` entities.
# MAGIC 
# MAGIC **5. NER Pipelines:**
# MAGIC 
# MAGIC > pipelines for all the available pretrained NER models.
# MAGIC 
# MAGIC **6. BERT Based NER Pipelines**
# MAGIC 
# MAGIC > pipelines for all the available Bert token classification models.
# MAGIC 
# MAGIC **7. ner_profiling_clinical and ner_profiling_biobert:**
# MAGIC 
# MAGIC > pipelines for exploring all the available pretrained NER models at once.
# MAGIC 
# MAGIC **8. ner_model_finder**
# MAGIC 
# MAGIC > a pipeline trained with bert embeddings that can be used to find the most appropriate NER model given the entity name.
# MAGIC 
# MAGIC **9. Resolver Pipelines**
# MAGIC 
# MAGIC > Pipelines for converting clinical entities to their UMLS CUI codes. You will just feed your text and it will return the corresponding UMLS codes.
# MAGIC 
# MAGIC **Also, you can find clinical CODE MAPPING pretrained pipelines in this notebook: [Healthcare Code Mapping Notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb)**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.explain_clinical_doc_carp 
# MAGIC 
# MAGIC a pipeline with ner_clinical, assertion_dl, re_clinical and ner_posology. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.

# COMMAND ----------

pipeline = PretrainedPipeline('explain_clinical_doc_carp', 'en', 'clinical/models')

# COMMAND ----------

pipeline.model.stages

# COMMAND ----------

# Load pretrained pipeline from local disk:

# >> pipeline_local = PretrainedPipeline.from_disk('/databricks/driver/explain_clinical_doc_carp_en_2.5.5_2.4_1597841630062')

# COMMAND ----------

text ="""A 28-year-old female with a history of gestational diabetes mellitus, used to take metformin 1000 mg two times a day, presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .
She was seen by the endocrinology service and discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals.
"""

annotations = pipeline.annotate(text)

annotations.keys()


# COMMAND ----------

import pandas as pd

rows = list(zip(annotations['tokens'], annotations['clinical_ner_tags'], annotations['posology_ner_tags'], annotations['pos_tags'], annotations['dependencies']))

df = pd.DataFrame(rows, columns = ['tokens','clinical_ner_tags','posology_ner_tags','POS_tags','dependencies'])

df.head(20)

# COMMAND ----------

text = 'Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain'

result = pipeline.fullAnnotate(text)[0]

chunks=[]
entities=[]
status=[]

for n,m in zip(result['clinical_ner_chunks'],result['assertion']):
    
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)
        
df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

df

# COMMAND ----------

text = """
The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also 
given 1 unit of Metformin daily.
He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 
12 units of insulin lispro with meals , and metformin 1000 mg two times a day.
"""

result = pipeline.fullAnnotate(text)[0]

chunks=[]
entities=[]
begins=[]
ends=[]

for n in result['posology_ner_chunks']:
    
    chunks.append(n.result)
    begins.append(n.begin)
    ends.append(n.end)
    entities.append(n.metadata['entity']) 
        
df = pd.DataFrame({'chunks':chunks, 'begin':begins, 'end':ends, 'entities':entities})

df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.   explain_clinical_doc_era
# MAGIC 
# MAGIC > a pipeline with `ner_clinical_events`, `assertion_dl` and `re_temporal_events_clinical`. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities.

# COMMAND ----------

era_pipeline = PretrainedPipeline('explain_clinical_doc_era', 'en', 'clinical/models')

# COMMAND ----------

era_pipeline.model.stages

# COMMAND ----------

text ="""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache.
She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 
12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. 
"""


result = era_pipeline.fullAnnotate(text)[0]


# COMMAND ----------

result.keys()

# COMMAND ----------

import pandas as pd

chunks=[]
entities=[]
begins=[]
ends=[]

for n in result['clinical_ner_chunks']:
    
    chunks.append(n.result)
    begins.append(n.begin)
    ends.append(n.end)
    entities.append(n.metadata['entity']) 
        
df = pd.DataFrame({'chunks':chunks, 'begin':begins, 'end':ends, 'entities':entities})

df

# COMMAND ----------

chunks=[]
entities=[]
status=[]

for n,m in zip(result['clinical_ner_chunks'],result['assertion']):
    
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    status.append(m.result)
        
df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})

df

# COMMAND ----------

import pandas as pd

def get_relations_df (results, col='relations'):
  rel_pairs=[]
  for rel in results[0][col]:
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

  rel_df.confidence = rel_df.confidence.astype(float)
  
  return rel_df

# COMMAND ----------

annotations = era_pipeline.fullAnnotate(text)

rel_df = get_relations_df (annotations, 'clinical_relations')

rel_df


# COMMAND ----------

annotations[0]['clinical_relations']

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.explain_clinical_doc_ade 
# MAGIC 
# MAGIC A pipeline for `Adverse Drug Events (ADE)` with `ner_ade_healthcare`, and `classifierdl_ade_biobert`. It will extract `ADE` and `DRUG` clinical entities, and then assign ADE status to a text(`True` means ADE, `False` means not related to ADE). Also extracts relations between `DRUG` and `ADE` entities (`1` means the adverse event and drug entities are related, `0` is not related).

# COMMAND ----------

ade_pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

# COMMAND ----------

result = ade_pipeline.fullAnnotate("I feel a bit drowsy & have a little blurred vision, so far no gastric problems.")

# COMMAND ----------

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

result = ade_pipeline.fullAnnotate(text)

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

light_result = ade_pipeline.fullAnnotate(text)[0]

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

text = """I have Rhuematoid Arthritis for 35 yrs and have been on many arthritis meds. 
I currently am on Relefen for inflamation, Prednisone 5mg, every other day and Enbrel injections once a week. 
I have no problems from these drugs. Eight months ago, another doctor put me on Lipitor 10mg daily because my chol was 240. 
Over a period of 6 months, it went down to 159, which was great, BUT I started having terrible aching pain in my arms about that time which was radiating down my arms from my shoulder to my hands.
"""
 
print (text)

results = ade_pipeline.fullAnnotate(text)

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
# MAGIC ## 4. Clinical Deidentification
# MAGIC 
# MAGIC This pipeline can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`, `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR` entities.

# COMMAND ----------

# MAGIC %md
# MAGIC **Clinical Deidentification Pretrained Pipeline List**
# MAGIC 
# MAGIC 
# MAGIC |index|model|lang|
# MAGIC |-----:|:-----|----|
# MAGIC | 1| [clinical_deidentification](https://nlp.johnsnowlabs.com/2022/03/03/clinical_deidentification_de_3_0.html)  |de|
# MAGIC | 2| [clinical_deidentification](https://nlp.johnsnowlabs.com/2021/05/27/clinical_deidentification_en.html)  |en|
# MAGIC | 3| [clinical_deidentification_glove](https://nlp.johnsnowlabs.com/2022/03/04/clinical_deidentification_glove_en_3_0.html)  |en|
# MAGIC | 4| [clinical_deidentification_glove_augmented](https://nlp.johnsnowlabs.com/2022/03/22/clinical_deidentification_glove_augmented_en_3_0.html)  |en|
# MAGIC | 5| [clinical_deidentification](https://nlp.johnsnowlabs.com/2022/03/02/clinical_deidentification_es_2_4.html)  |es|
# MAGIC | 6| [clinical_deidentification_augmented](https://nlp.johnsnowlabs.com/2022/03/03/clinical_deidentification_augmented_es_2_4.html)  |es|
# MAGIC | 7| [clinical_deidentification](https://nlp.johnsnowlabs.com/2022/03/04/clinical_deidentification_fr_2_4.html)  |fr|
# MAGIC | 8| [clinical_deidentification](https://nlp.johnsnowlabs.com/2022/03/28/clinical_deidentification_it_3_0.html)  |it|
# MAGIC | 9| [clinical_deidentification](https://nlp.johnsnowlabs.com/2022/06/21/clinical_deidentification_pt_3_0.html)  |pt|
# MAGIC | 10| [clinical_deidentification](https://nlp.johnsnowlabs.com/2022/06/28/clinical_deidentification_ro_3_0.html)  |ro|

# COMMAND ----------

# MAGIC %md
# MAGIC You can find German, Spanish, French, Italian, Portuguese and Romanian deidentification models and pretrained pipeline examples in this notebook:   [Clinical Multi Language Deidentification Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.1.Clinical_Multi_Language_Deidentification.ipynb)

# COMMAND ----------

deid_pipeline = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")

# COMMAND ----------

deid_res = deid_pipeline.annotate("Record date : 2093-01-13 , David Hale , M.D .  Name : Hendrickson , Ora MR 25 years-old . # 719435 Date : 01/13/93 . Signed by Oliveira Sander . Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street. Phone 302-786-5227.")

# COMMAND ----------

deid_res.keys()

# COMMAND ----------

pd.set_option("display.max_colwidth", 100)

df= pd.DataFrame(list(zip(deid_res["sentence"], 
                          deid_res["masked"],
                          deid_res["masked_with_chars"], 
                          deid_res["masked_fixed_length_chars"], 
                          deid_res["obfuscated"])),
                 columns= ["Sentence", "Masked", "Masked with Chars", "Masked with Fixed Chars", "Obfuscated"])

df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.NER Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC **`NER pretrained ` Model List**
# MAGIC 
# MAGIC |index|model|index|model|index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|-----:|:-----|-----:|:-----|
# MAGIC | 1| [jsl_ner_wip_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/jsl_ner_wip_clinical_pipeline_en_3_0.html)  | 2| [jsl_ner_wip_greedy_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/jsl_ner_wip_greedy_biobert_pipeline_en_3_0.html)  | 3| [jsl_ner_wip_greedy_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/jsl_ner_wip_greedy_clinical_pipeline_en_3_0.html)  | 4| [jsl_ner_wip_modifier_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/jsl_ner_wip_modifier_clinical_pipeline_en_3_0.html)  |
# MAGIC | 5| [jsl_rd_ner_wip_greedy_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/jsl_rd_ner_wip_greedy_biobert_pipeline_en_3_0.html)  | 6| [jsl_rd_ner_wip_greedy_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/jsl_rd_ner_wip_greedy_clinical_pipeline_en_3_0.html)  | 7| [ner_abbreviation_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_abbreviation_clinical_pipeline_en_3_0.html)  | 8| [ner_ade_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_ade_biobert_pipeline_en_3_0.html)  |
# MAGIC | 9| [ner_ade_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_ade_clinical_pipeline_en_3_0.html)  | 10| [ner_ade_clinicalbert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_ade_clinicalbert_pipeline_en_3_0.html)  | 11| [ner_ade_healthcare_pipeline](https://nlp.johnsnowlabs.com/2022/03/22/ner_ade_healthcare_pipeline_en_3_0.html)  | 12| [ner_anatomy_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_anatomy_biobert_pipeline_en_3_0.html)  |
# MAGIC | 13| [ner_anatomy_coarse_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_anatomy_coarse_biobert_pipeline_en_3_0.html)  | 14| [ner_anatomy_coarse_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_anatomy_coarse_pipeline_en_3_0.html)  | 15| [ner_anatomy_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_anatomy_pipeline_en_3_0.html)  | 16| [ner_bacterial_species_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_bacterial_species_pipeline_en_3_0.html)  |
# MAGIC | 17| [ner_biomarker_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_biomarker_pipeline_en_3_0.html)  | 18| [ner_biomedical_bc2gm_pipeline](https://nlp.johnsnowlabs.com/2022/06/22/ner_biomedical_bc2gm_pipeline_en_3_0.html)  | 19| [ner_bionlp_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_bionlp_biobert_pipeline_en_3_0.html)  | 20| [ner_bionlp_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_bionlp_pipeline_en_3_0.html)  |
# MAGIC | 21| [ner_cancer_genetics_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_cancer_genetics_pipeline_en_3_0.html)  | 22| [ner_cellular_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_cellular_biobert_pipeline_en_3_0.html)  | 23| [ner_cellular_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_cellular_pipeline_en_3_0.html)  | 24| [ner_chemicals_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_chemicals_pipeline_en_3_0.html)  |
# MAGIC | 25| [ner_chemprot_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_chemprot_biobert_pipeline_en_3_0.html)  | 26| [ner_chemprot_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_chemprot_clinical_pipeline_en_3_0.html)  | 27| [ner_chexpert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_chexpert_pipeline_en_3_0.html)  | 28| [ner_clinical_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_clinical_biobert_pipeline_en_3_0.html)  |
# MAGIC | 29| [ner_clinical_large_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_clinical_large_pipeline_en_3_0.html)  | 30| [ner_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_clinical_pipeline_en_3_0.html)  | 31| [ner_clinical_trials_abstracts_pipeline](https://nlp.johnsnowlabs.com/2022/06/27/ner_clinical_trials_abstracts_pipeline_en_3_0.html)  | 32| [ner_diseases_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_diseases_biobert_pipeline_en_3_0.html)  |
# MAGIC | 33| [ner_diseases_large_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_diseases_large_pipeline_en_3_0.html)  | 34| [ner_diseases_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_diseases_pipeline_en_3_0.html)  | 35| [ner_drugprot_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_drugprot_clinical_pipeline_en_3_0.html)  | 36| [ner_drugs_greedy_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_drugs_greedy_pipeline_en_3_0.html)  |
# MAGIC | 37| [ner_drugs_large_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_drugs_large_pipeline_en_3_0.html)  | 38| [ner_drugs_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_drugs_pipeline_en_3_0.html)  | 39| [ner_events_admission_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_events_admission_clinical_pipeline_en_3_0.html)  | 40| [ner_events_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_events_biobert_pipeline_en_3_0.html)  |
# MAGIC | 41| [ner_events_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_events_clinical_pipeline_en_3_0.html)  | 42| [ner_events_healthcare_pipeline](https://nlp.johnsnowlabs.com/2022/03/22/ner_events_healthcare_pipeline_en_3_0.html)  | 43| [ner_genetic_variants_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_genetic_variants_pipeline_en_3_0.html)  | 44| [ner_healthcare_pipeline](https://nlp.johnsnowlabs.com/2022/03/22/ner_healthcare_pipeline_en_3_0.html)  |
# MAGIC | 45| [ner_human_phenotype_gene_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_human_phenotype_gene_biobert_pipeline_en_3_0.html)  | 46| [ner_human_phenotype_gene_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_human_phenotype_gene_clinical_pipeline_en_3_0.html)  | 47| [ner_human_phenotype_go_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_human_phenotype_go_biobert_pipeline_en_3_0.html)  | 48| [ner_human_phenotype_go_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_human_phenotype_go_clinical_pipeline_en_3_0.html)  |
# MAGIC | 49| [ner_jsl_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_jsl_biobert_pipeline_en_3_0.html)  | 50| [ner_jsl_enriched_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_jsl_enriched_biobert_pipeline_en_3_0.html)  | 51| [ner_jsl_enriched_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_jsl_enriched_pipeline_en_3_0.html)  | 52| [ner_jsl_greedy_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_jsl_greedy_biobert_pipeline_en_3_0.html)  |
# MAGIC | 53| [ner_jsl_greedy_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_jsl_greedy_pipeline_en_3_0.html)  | 54| [ner_jsl_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_jsl_pipeline_en_3_0.html)  | 55| [ner_jsl_slim_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_jsl_slim_pipeline_en_3_0.html)  | 56| [ner_measurements_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_measurements_clinical_pipeline_en_3_0.html)  |
# MAGIC | 57| [ner_medmentions_coarse_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_medmentions_coarse_pipeline_en_3_0.html)  | 58| [ner_nihss_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_nihss_pipeline_en_3_0.html)  | 59| [ner_pathogen_pipeline](https://nlp.johnsnowlabs.com/2022/06/29/ner_pathogen_pipeline_en_3_0.html)  | 60| [ner_posology_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_posology_biobert_pipeline_en_3_0.html)  |
# MAGIC | 61| [ner_posology_experimental_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_posology_experimental_pipeline_en_3_0.html)  | 62| [ner_posology_greedy_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_posology_greedy_pipeline_en_3_0.html)  | 63| [ner_posology_healthcare_pipeline](https://nlp.johnsnowlabs.com/2022/03/22/ner_posology_healthcare_pipeline_en_3_0.html)  | 64| [ner_posology_large_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_posology_large_biobert_pipeline_en_3_0.html)  |
# MAGIC | 65| [ner_posology_large_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_posology_large_pipeline_en_3_0.html)  | 66| [ner_posology_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_posology_pipeline_en_3_0.html)  | 67| [ner_posology_small_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_posology_small_pipeline_en_3_0.html)  | 68| [ner_radiology_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_radiology_pipeline_en_3_0.html)  |
# MAGIC | 69| [ner_radiology_wip_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_radiology_wip_clinical_pipeline_en_3_0.html)  | 70| [ner_risk_factors_biobert_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_risk_factors_biobert_pipeline_en_3_0.html)  | 71| [ner_risk_factors_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/ner_risk_factors_pipeline_en_3_0.html)  | 72| [ner_medication_pipeline](https://nlp.johnsnowlabs.com/2022/07/26/ner_medication_pipeline_en_3_0.html)|

# COMMAND ----------

# MAGIC %md
# MAGIC **Let's show an example of `ner_jsl_pipeline` can label clinical entities with about 80 different labels.**

# COMMAND ----------

ner_pipeline = PretrainedPipeline('ner_jsl_pipeline', 'en', 'clinical/models')

# COMMAND ----------

ner_pipeline.model.stages

# COMMAND ----------

text = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . 
Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . 
Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . 
The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . 
The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . 
Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . 
It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge ."""

greedy_result = ner_pipeline.fullAnnotate(text)[0]

# COMMAND ----------

greedy_result.keys()

# COMMAND ----------

import pandas as pd

chunks=[]
entities=[]
begins=[]
ends=[]

for n in greedy_result['ner_chunk']:
    
    chunks.append(n.result)
    begins.append(n.begin)
    ends.append(n.end)
    entities.append(n.metadata['entity']) 
        
df = pd.DataFrame({'chunks':chunks, 'begin':begins, 'end':ends, 'entities':entities})

df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.Bert Based NER Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC **`bert token classification pretrained ` Model List**
# MAGIC 
# MAGIC 
# MAGIC |index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|
# MAGIC | 1| [bert_token_classifier_drug_development_trials_pipeline](https://nlp.johnsnowlabs.com/2022/03/23/bert_token_classifier_drug_development_trials_pipeline_en_3_0.html)  | 8| [bert_token_classifier_ner_chemprot_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_chemprot_pipeline_en_3_0.html)  |
# MAGIC | 2| [bert_token_classifier_ner_ade_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_ade_pipeline_en_3_0.html)  | 9| [bert_token_classifier_ner_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_clinical_pipeline_en_3_0.html)  |
# MAGIC | 3| [bert_token_classifier_ner_anatomy_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_anatomy_pipeline_en_3_0.html)  | 10| [bert_token_classifier_ner_deid_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_deid_pipeline_en_3_0.html)  |
# MAGIC | 4| [bert_token_classifier_ner_bacteria_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_bacteria_pipeline_en_3_0.html)  | 11| [bert_token_classifier_ner_drugs_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_drugs_pipeline_en_3_0.html)  |
# MAGIC | 5| [bert_token_classifier_ner_bionlp_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_bionlp_pipeline_en_3_0.html)  | 12| [bert_token_classifier_ner_jsl_pipeline](https://nlp.johnsnowlabs.com/2022/03/23/bert_token_classifier_ner_jsl_pipeline_en_3_0.html)  |
# MAGIC | 6| [bert_token_classifier_ner_cellular_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_cellular_pipeline_en_3_0.html)  | 13| [bert_token_classifier_ner_jsl_slim_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_jsl_slim_pipeline_en_3_0.html)  |
# MAGIC | 7| [bert_token_classifier_ner_chemicals_pipeline](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_chemicals_pipeline_en_3_0.html)  |

# COMMAND ----------

# MAGIC %md
# MAGIC **Let's show an example of `bert_token_classifier_ner_drugs_pipeline` can extract `DRUG` entities in clinical texts.**

# COMMAND ----------

bert_token_pipeline = PretrainedPipeline("bert_token_classifier_ner_drugs_pipeline", "en", "clinical/models")

# COMMAND ----------

bert_token_pipeline.model.stages

# COMMAND ----------

test_sentence = """The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes. With the objective of determining the usefulnessof vinorelbine monotherapy in patients with advanced or recurrent breast cancerafter standard therapy, we evaluated the efficacy and safety of vinorelbine inpatients previously treated with anthracyclines and taxanes."""

bert_result = bert_token_pipeline.fullAnnotate(test_sentence)[0]

# COMMAND ----------

bert_result.keys()

# COMMAND ----------

bert_result["ner_chunk"][0]

# COMMAND ----------

import pandas as pd

chunks=[]
entities=[]
begins=[]
ends=[]

for n in bert_result['ner_chunk']:
    
    chunks.append(n.result)
    begins.append(n.begin)
    ends.append(n.end)
    entities.append(n.metadata['entity']) 
        
df = pd.DataFrame({'chunks':chunks, 'begin':begins, 'end':ends, 'entities':entities})

df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.NER Profiling Pipelines
# MAGIC 
# MAGIC We can use pretrained NER profiling pipelines for exploring all the available pretrained NER models at once. In Spark NLP we have two different NER profiling pipelines;
# MAGIC 
# MAGIC - `ner_profiling_clinical` : Returns results for clinical NER models trained with `embeddings_clinical`.
# MAGIC - `ner_profiling_biobert` : Returns results for clinical NER models trained with `biobert_pubmed_base_cased`.
# MAGIC 
# MAGIC For more examples, please check [this notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.2.Pretrained_NER_Profiling_Pipelines.ipynb).

# COMMAND ----------

# MAGIC %md
# MAGIC <center><b>NER Profiling Clinical Model List</b>
# MAGIC 
# MAGIC || | | |
# MAGIC |--------------|-----------------|-----------------|-----------------|
# MAGIC | jsl_ner_wip_clinical | jsl_ner_wip_greedy_clinical | jsl_ner_wip_modifier_clinical | jsl_rd_ner_wip_greedy_clinical |
# MAGIC | ner_abbreviation_clinical | ner_ade_binary | ner_ade_clinical | ner_anatomy |
# MAGIC | ner_anatomy_coarse | ner_bacterial_species | ner_biomarker | ner_biomedical_bc2gm |
# MAGIC | ner_bionlp | ner_cancer_genetics | ner_cellular | ner_chemd_clinical |
# MAGIC | ner_chemicals | ner_chemprot_clinical | ner_chexpert | ner_clinical |
# MAGIC | ner_clinical_large | ner_clinical_trials_abstracts | ner_covid_trials | ner_deid_augmented |
# MAGIC | ner_deid_enriched | ner_deid_generic_augmented | ner_deid_large | ner_deid_sd |
# MAGIC | ner_deid_sd_large | ner_deid_subentity_augmented | ner_deid_subentity_augmented_i2b2 | ner_deid_synthetic |
# MAGIC | ner_deidentify_dl | ner_diseases | ner_diseases_large | ner_drugprot_clinical |
# MAGIC | ner_drugs | ner_drugs_greedy | ner_drugs_large | ner_events_admission_clinical |
# MAGIC | ner_events_clinical | ner_genetic_variants | ner_human_phenotype_gene_clinical | ner_human_phenotype_go_clinical |
# MAGIC | ner_jsl | ner_jsl_enriched | ner_jsl_greedy | ner_jsl_slim |
# MAGIC | ner_living_species | ner_measurements_clinical | ner_medmentions_coarse | ner_nature_nero_clinical |
# MAGIC | ner_nihss | ner_pathogen | ner_posology | ner_posology_experimental |
# MAGIC | ner_posology_greedy | ner_posology_large | ner_posology_small | ner_radiology |
# MAGIC | ner_radiology_wip_clinical | ner_risk_factors | ner_supplement_clinical | nerdl_tumour_demo |
# MAGIC 
# MAGIC <b>NER Profiling BioBert Model List</b>
# MAGIC 
# MAGIC | | |
# MAGIC |-|-|
# MAGIC | ner_cellular_biobert           | ner_clinical_biobert             |
# MAGIC | ner_diseases_biobert           | ner_anatomy_coarse_biobert       |
# MAGIC | ner_events_biobert             | ner_human_phenotype_gene_biobert |
# MAGIC | ner_bionlp_biobert             | ner_posology_large_biobert       |
# MAGIC | ner_jsl_greedy_biobert         | jsl_rd_ner_wip_greedy_biobert    |
# MAGIC | ner_jsl_biobert                | ner_posology_biobert             |
# MAGIC | ner_anatomy_biobert            | jsl_ner_wip_greedy_biobert       |
# MAGIC | ner_jsl_enriched_biobert       | ner_chemprot_biobert             |
# MAGIC | ner_human_phenotype_go_biobert | ner_ade_biobert                  |
# MAGIC | ner_deid_biobert               | ner_risk_factors_biobert         |
# MAGIC | ner_deid_enriched_biobert      | ner_living_species_biobert                                |
# MAGIC 
# MAGIC 
# MAGIC </center>
# MAGIC 
# MAGIC You can check [Models Hub](https://nlp.johnsnowlabs.com/models) page for more information about all these models and more.

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

clinical_profiling_pipeline = PretrainedPipeline("ner_profiling_clinical", "en", "clinical/models")

# COMMAND ----------

text = '''A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .'''

# COMMAND ----------

clinical_result = clinical_profiling_pipeline.fullAnnotate(text)[0]
clinical_result.keys()

# COMMAND ----------

import pandas as pd

def get_token_results(light_result):

  tokens = [j.result for j in light_result["token"]]
  sentences = [j.metadata["sentence"] for j in light_result["token"]]
  begins = [j.begin for j in light_result["token"]]
  ends = [j.end for j in light_result["token"]]
  model_list = [ a for a in light_result.keys() if (a not in ["sentence", "token"] and "_chunks" not in a)]

  df = pd.DataFrame({'sentence':sentences, 'begin': begins, 'end': ends, 'token':tokens})

  for model_name in model_list:
    
    temp_df = pd.DataFrame(light_result[model_name])
    temp_df["jsl_label"] = temp_df.iloc[:,0].apply(lambda x : x.result)
    temp_df = temp_df[["jsl_label"]]

    # temp_df = get_ner_result(model_name)
    temp_df.columns = [model_name]
    df = pd.concat([df, temp_df], axis=1)
    
  return df

# COMMAND ----------

get_token_results(clinical_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.NER Model Finder Pretrained Pipeline
# MAGIC `ner_model_finder`  pretrained pipeline trained with bert embeddings that can be used to find the most appropriate NER model given the entity name.

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline
finder_pipeline = PretrainedPipeline("ner_model_finder", "en", "clinical/models")

# COMMAND ----------

result = finder_pipeline.fullAnnotate("oncology")[0]
result.keys()

# COMMAND ----------

# MAGIC %md
# MAGIC From the metadata in the 'model_names' column, we'll get to the top models to the given 'oncology' entity and oncology related categories.

# COMMAND ----------

df= pd.DataFrame(zip(result["model_names"][0].metadata["all_k_resolutions"].split(":::"), 
                     result["model_names"][0].metadata["all_k_results"].split(":::")), 
                 columns=["category", "top_models"])

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.Resolver Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC We have **Resolver pipelines** for converting clinical entities to their UMLS CUI codes. You will just feed your text and it will return the corresponding UMLS codes.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Resolver Pipelines Model List:** 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC | Pipeline Name                                                                                                                            | Entity                  | Target   |
# MAGIC |------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|----------|
# MAGIC | [umls_drug_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/26/umls_drug_resolver_pipeline_en_3_0.html)                           | Drugs                   | UMLS CUI |
# MAGIC | [umls_clinical_findings_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/26/umls_clinical_findings_resolver_pipeline_en_3_0.html) | Clinical Findings       | UMLS CUI |
# MAGIC | [umls_disease_syndrome_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/26/umls_disease_syndrome_resolver_pipeline_en_3_0.html)   | Disease and Syndromes   | UMLS CUI |
# MAGIC | [umls_major_concepts_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/25/umls_major_concepts_resolver_pipeline_en_3_0.html)       | Clinical Major Concepts | UMLS CUI |
# MAGIC | [umls_drug_substance_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/07/25/umls_drug_substance_resolver_pipeline_en_3_0.html)       | Drug Substances         | UMLS CUI |
# MAGIC | [medication_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/09/01/medication_resolver_pipeline_en.html)                           | Drugs                   | RxNorm, UMLS, NDC, SNOMED CT |
# MAGIC | [medication_resolver_transform_pipeline](https://nlp.johnsnowlabs.com/2022/09/01/medication_resolver_transform_pipeline_en.html)                           | Drugs                   | RxNorm, UMLS, NDC, SNOMED CT |
# MAGIC | [icd9_resolver_pipeline](https://nlp.johnsnowlabs.com/2022/09/30/icd9_resolver_pipeline_en.html)                           | PROBLEM                   | ICD-9-CM |

# COMMAND ----------

pipeline= PretrainedPipeline("umls_clinical_findings_resolver_pipeline", "en", "clinical/models")
result= pipeline.fullAnnotate("HTG-induced pancreatitis associated with an acute hepatitis, and obesity")[0]

# COMMAND ----------

import pandas as pd

chunks=[]
entities=[]
resolver= []

for n, m in list(zip(result['chunk'], result["umls"])):
    
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    resolver.append(m.result)
        
df = pd.DataFrame({'chunks':chunks, 'ner_label':entities, 'umls_code': resolver})

df