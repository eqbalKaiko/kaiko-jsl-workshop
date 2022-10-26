# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # ONCOLOGY MODELS

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook includes details about different kinds of pretrained models to extract oncology-related information from clinical texts, together with examples of each type of model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

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
from sparknlp_jsl.pretrained import InternalResourceDownloader

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
# MAGIC ## List of Pretrained Models

# COMMAND ----------

df = pd.DataFrame()
for model_type in ['MedicalNerModel', 'BertForTokenClassification', 'RelationExtractionModel', 'RelationExtractionDLModel', 'AssertionDLModel']:
    model_list = list(set([model[0] for model in InternalResourceDownloader.returnPrivateModels(model_type) if 'oncology' in model[0]]))
    if len(model_list) > 0:
      df = pd.concat([df, pd.DataFrame(model_list, columns = [model_type])], axis = 1)
    
df.fillna('')

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER Models

# COMMAND ----------

# MAGIC %md
# MAGIC The NER models from the list include different entity groups and levels of granularity. If you want to extract as much information as possible from oncology texts, then ner_oncology_wip is the best option for you, as it is the most general and granular model. But you may want to use other models depending on your needs (for instance, if you need to extract information related with staging, ner_oncology_tnm_wip would be the most suitable model).

# COMMAND ----------

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
        
sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
 
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setSplitChars(["-", "\/"])

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
    .setInputCols(["sentence","token"])\
    .setOutputCol("embeddings")

# ner_oncology_wip

ner_oncology_wip = MedicalNerModel.pretrained("ner_oncology_wip","en","clinical/models")\
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner_oncology_wip")\

ner_oncology_wip_converter = NerConverter()\
    .setInputCols(["sentence","token","ner_oncology_wip"])\
    .setOutputCol("ner_oncology_wip_chunk")

# ner_oncology_tnm_wip

ner_oncology_tnm_wip = MedicalNerModel.pretrained("ner_oncology_tnm_wip","en","clinical/models")\
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner_oncology_tnm_wip")\

ner_oncology_tnm_wip_converter = NerConverter()\
    .setInputCols(["sentence","token","ner_oncology_tnm_wip"])\
    .setOutputCol("ner_oncology_tnm_wip_chunk")

# # ner_oncology_biomarker_wip

ner_oncology_biomarker_wip = MedicalNerModel.pretrained("ner_oncology_biomarker_wip","en","clinical/models")\
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner_oncology_biomarker_wip")\

ner_oncology_biomarker_wip_converter = NerConverter()\
    .setInputCols(["sentence","token","ner_oncology_biomarker_wip"])\
    .setOutputCol("ner_oncology_biomarker_wip_chunk")

ner_stages = [document_assembler,
    sentence_detector,
    tokenizer,
    word_embeddings,
    ner_oncology_wip,
    ner_oncology_wip_converter,
    ner_oncology_tnm_wip,
    ner_oncology_tnm_wip_converter,
    ner_oncology_biomarker_wip,
    ner_oncology_biomarker_wip_converter]

ner_pipeline = Pipeline(stages=ner_stages)

empty_data = spark.createDataFrame([[""]]).toDF("text")

ner_model = ner_pipeline.fit(empty_data)

# COMMAND ----------

ner_oncology_wip_labels = list(set([label.split('-')[-1] for label in ner_oncology_wip.getClasses() if label != 'O']))

len(ner_oncology_wip_labels)

# COMMAND ----------

label_df = pd.DataFrame()
for column in range((len(ner_oncology_wip_labels)//10)+1):
  label_df = pd.concat([label_df, pd.DataFrame(ner_oncology_wip_labels, columns = [''])[column*10:(column+1)*10].reset_index(drop= True)], axis = 1)

label_df.fillna('')

# COMMAND ----------

ner_oncology_tnm_wip_labels = list(set([label.split('-')[-1] for label in ner_oncology_tnm_wip.getClasses() if label != 'O']))

print(ner_oncology_tnm_wip_labels)

# COMMAND ----------

ner_oncology_biomarker_wip_labels = list(set([label.split('-')[-1] for label in ner_oncology_biomarker_wip.getClasses() if label != 'O']))

print(ner_oncology_biomarker_wip_labels)

# COMMAND ----------

sample_text_1 = '''A 65-year-old woman had a history of debulking surgery, bilateral oophorectomy with omentectomy, total anterior hysterectomy with radical pelvic lymph nodes dissection due to ovarian carcinoma (mucinous-type carcinoma, stage Ic) 1 year ago. Patient's medical compliance was poor and failed to complete her chemotherapy (cyclophosphamide 750 mg/m2, carboplatin 300 mg/m2). Recently, she noted a palpable right breast mass, 15 cm in size which nearly occupied the whole right breast in 2 months. Core needle biopsy revealed metaplastic carcinoma. Neoadjuvant chemotherapy with the regimens of Taxotere (75 mg/m2), Epirubicin (75 mg/m2), and Cyclophosphamide (500 mg/m2) was given for 6 cycles with poor response, followed by a modified radical mastectomy (MRM) with dissection of axillary lymph nodes and skin grafting. Postoperatively, radiotherapy was done with 5000 cGy in 25 fractions. The histopathologic examination revealed a metaplastic carcinoma with squamous differentiation associated with adenomyoepithelioma. Immunohistochemistry study showed that the tumor cells are positive for epithelial markers-cytokeratin (AE1/AE3) stain, and myoepithelial markers, including cytokeratin 5/6 (CK 5/6), p63, and S100 stains. Expressions of hormone receptors, including ER, PR, and Her-2/Neu, were all negative. The dissected axillary lymph nodes showed metastastic carcinoma with negative hormone receptors in 3 nodes. The patient was staged as pT3N1aM0, with histologic tumor grade III.'''

sample_text_2 = '''She underwent a computed tomography (CT) scan of the abdomen and pelvis, which showed a complex ovarian mass. A Pap smear performed one month later was positive for atypical glandular cells suspicious for adenocarcinoma. The pathologic specimen showed extension of the tumor throughout the fallopian tubes, appendix, omentum, and 5 out of 5 enlarged lymph nodes. The final pathologic diagnosis of the tumor was stage IIIC papillary serous ovarian adenocarcinoma. Two months later, the patient was diagnosed with lung metastases.'''

sample_text_3 = '''In the bone- marrow (BM) aspiration, blasts accounted for 88.1% of ANCs, which were positive for CD9, CD10, CD13, CD19, CD20, CD34, CD38, CD58, CD66c, CD123, HLA-DR, cCD79a, and TdT on flow cytometry.

Measurements of serum tumor markers showed elevated level of cytokeratin 19 fragment (Cyfra21-1: 4.77 ng/mL), neuron-specific enolase (NSE: 19.60 ng/mL), and squamous cell carcinoma antigen (SCCA: 2.58 ng/mL). The results were negative for serum carbohydrate antigen 125 (CA125), carcinoembryonic antigen (CEA) and vascular endothelial growth factor (VEGF). Immunohistochemical staining showed positive staining for CK5/6, P40 and PD-L1 (+ 80% tumor cells), and negative staining for TTF-1, PD-1 and weakly positive staining for ALK. Molecular analysis indicated no EGFR mutation or ROS1 fusion.'''

# COMMAND ----------

data = spark.createDataFrame(pd.DataFrame([sample_text_1, sample_text_2, sample_text_3], columns = ['text']))

# COMMAND ----------

results = ner_model.transform(data).collect()

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

# COMMAND ----------

ner_html_1 = visualiser.display(results[0], label_col='ner_oncology_wip_chunk', return_html=True)

displayHTML(ner_html_1)

# COMMAND ----------

ner_html_2 = visualiser.display(results[1], label_col='ner_oncology_tnm_wip_chunk', return_html=True)

displayHTML(ner_html_2)

# COMMAND ----------

ner_html_3 = visualiser.display(results[2], label_col='ner_oncology_biomarker_wip_chunk', return_html=True)

displayHTML(ner_html_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Relation Extraction Models

# COMMAND ----------

# MAGIC %md
# MAGIC RE Models are used to link entities that are related. For oncology entities, you can use general models (such as re_oncology_granular_wip) or you can select a specific model depending on your needs (e.g. re_oncology_size_wip to link tumors and their sizes, or re_oncology_biomarker_result_wip to link biomarkers and their results).

# COMMAND ----------

pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos_tags", "token"]) \
    .setOutputCol("dependencies")

re_oncology_granular_wip = RelationExtractionModel.pretrained("re_oncology_granular_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_oncology_wip_chunk", "dependencies"]) \
    .setOutputCol("re_oncology_granular_wip") \
    .setRelationPairs(['Date-Cancer_Dx', 'Cancer_Dx-Date', 'Tumor_Finding-Site_Breast', 'Site_Breast-Tumor_Finding',
                       'Relative_Date-Tumor_Finding', 'Tumor_Fiding-Relative_Date', 'Tumor_Finding-Tumor_Size', 'Tumor_Size-Tumor_Finding',
                       'Pathology_Test-Cancer_Dx', 'Cancer_Dx-Pathology_Test']) \
    .setMaxSyntacticDistance(10)    

re_oncology_size_wip = RelationExtractionModel.pretrained("re_oncology_size_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_oncology_wip_chunk", "dependencies"]) \
    .setOutputCol("re_oncology_size_wip") \
    .setRelationPairs(['Tumor_Finding-Tumor_Size', 'Tumor_Size-Tumor_Finding']) \
    .setMaxSyntacticDistance(10)    

re_oncology_biomarker_result_wip = RelationExtractionModel.pretrained("re_oncology_biomarker_result_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_oncology_biomarker_wip_chunk", "dependencies"]) \
    .setOutputCol("re_oncology_biomarker_result_wip") \
    .setRelationPairs(['Biomarker-Biomarker_Result', 'Biomarker_Result-Biomarker']) \
    .setMaxSyntacticDistance(10)      

re_stages = ner_stages + [pos_tagger, dependency_parser, re_oncology_granular_wip, re_oncology_size_wip, re_oncology_biomarker_result_wip]

re_pipeline = Pipeline(stages=re_stages)

re_model = re_pipeline.fit(empty_data)

# COMMAND ----------

sample_text_4 = '''Two years ago, she noted a palpable right breast mass, 15 cm in size. Core needle biopsy revealed metaplastic carcinoma.'''

sample_text_5 = '''The patient presented a 2 cm mass in her left breast, and the tumor in her other breast was 3 cm long.'''

sample_text_6 = '''Immunohistochemical staining showed positive staining for CK5/6, P40 and PD-L1, and negative staining for TTF-1, PD-1 and weakly positive staining for ALK. Immunohistochemistry study showed that the tumor cells are positive for epithelial markers-cytokeratin and myoepithelial markers, including cytokeratin 5/6, p63, and S100 stains.'''

# COMMAND ----------

re_data = spark.createDataFrame(pd.DataFrame([sample_text_4, sample_text_5, sample_text_6], columns = ['text']))

# COMMAND ----------

re_results = re_model.transform(re_data).collect()

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer

re_visualiser = RelationExtractionVisualizer()

# COMMAND ----------

re_html_1 = re_visualiser.display(re_results[0], relation_col='re_oncology_granular_wip', return_html=True)

displayHTML(re_html_1)

# COMMAND ----------

re_html_2 = re_visualiser.display(re_results[1], relation_col='re_oncology_size_wip', return_html=True)

displayHTML(re_html_2)

# COMMAND ----------

re_html_3 = re_visualiser.display(re_results[2], relation_col='re_oncology_biomarker_result_wip', return_html=True)

displayHTML(re_html_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assertion Status Models

# COMMAND ----------

# MAGIC %md
# MAGIC With assertion status models, you will be able to identify if entities included in texts are mentioned as something present, absent, hypothetical, possible, etc. You can either try using the general assertion_oncology_wip model, or other models that are recommended for specific entity groups (such as assertion_oncology_problem_wip, which should be used for problem entities like Cancer_Dx or Metastasis).

# COMMAND ----------

assertion_oncology_wip = AssertionDLModel.pretrained("assertion_oncology_wip", "en", "clinical/models") \
    .setInputCols(["sentence", 'ner_oncology_wip_chunk', "embeddings"]) \
    .setOutputCol("assertion_oncology_wip")

assertion_oncology_problem_wip = AssertionDLModel.pretrained("assertion_oncology_problem_wip", "en", "clinical/models") \
    .setInputCols(["sentence", 'ner_oncology_tnm_wip_chunk', "embeddings"]) \
    .setOutputCol("assertion_oncology_problem_wip")

assertion_oncology_treatment_binary_wip = AssertionDLModel.pretrained("assertion_oncology_treatment_binary_wip", "en", "clinical/models") \
    .setInputCols(["sentence", 'ner_oncology_wip_chunk', "embeddings"]) \
    .setOutputCol("assertion_oncology_treatment_binary_wip")

assertion_stages = ner_stages + [assertion_oncology_wip, assertion_oncology_problem_wip, assertion_oncology_treatment_binary_wip]

assertion_pipeline = Pipeline(stages=assertion_stages)

assertion_model = assertion_pipeline.fit(empty_data)

# COMMAND ----------

sample_text_7 = 'The patient is suspected to have colorectal cancer. Family history is positive for other cancers. The result of the biopsy was positive. A CT scan was ordered to rule out metastases.'

sample_text_8 = 'The patient was diagnosed with breast cancer. She was suspected to have metastases in her lungs. Her family history is positive for ovarian cancer.'

sample_text_9 = 'The patient underwent a mastectomy. We recommend to start radiotherapy. The patient refused to chemotherapy.'

# COMMAND ----------

assertion_data = spark.createDataFrame(pd.DataFrame([sample_text_7, sample_text_8, sample_text_9], columns = ['text']))

# COMMAND ----------

assertion_results = assertion_model.transform(assertion_data).collect()

# COMMAND ----------

from sparknlp_display import AssertionVisualizer

assertion_visualiser = AssertionVisualizer()

# COMMAND ----------

assert_html_1 = assertion_visualiser.display(assertion_results[0], label_col ='ner_oncology_wip_chunk', assertion_col='assertion_oncology_wip', return_html=True)

displayHTML(assert_html_1)

# COMMAND ----------

assert_html_2 = assertion_visualiser.display(assertion_results[1], label_col ='ner_oncology_tnm_wip_chunk', assertion_col='assertion_oncology_problem_wip', return_html=True)

displayHTML(assert_html_2)

# COMMAND ----------

assert_html_3 = assertion_visualiser.display(assertion_results[2], label_col ='ner_oncology_wip_chunk', assertion_col='assertion_oncology_treatment_binary_wip', return_html=True)

displayHTML(assert_html_3)