# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Training and Reusing Clinical Named Entity Recognition Models

# COMMAND ----------

# MAGIC %md
# MAGIC Please make sure that your cluster is setup properly according to https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-spark-nlp-for-healthcare-on-databricks

# COMMAND ----------

# MAGIC %md
# MAGIC **NER Model Implementation in Spark NLP**
# MAGIC 
# MAGIC   The deep neural network architecture for NER model in
# MAGIC Spark NLP is BiLSTM-CNN-Char framework. a slightly modified version of the architecture proposed by Jason PC Chiu and Eric Nichols ([Named Entity Recognition with Bidirectional LSTM-CNNs
# MAGIC ](https://arxiv.org/abs/1511.08308)). It is a neural network architecture that
# MAGIC automatically detects word and character-level features using a
# MAGIC hybrid bidirectional LSTM and CNN architecture, eliminating
# MAGIC the need for most feature engineering steps.
# MAGIC 
# MAGIC   In the original framework, the CNN extracts a fixed length
# MAGIC feature vector from character-level features. For each word,
# MAGIC these vectors are concatenated and fed to the BLSTM network
# MAGIC and then to the output layers. They employed a stacked
# MAGIC bi-directional recurrent neural network with long short-term
# MAGIC memory units to transform word features into named entity
# MAGIC tag scores. The extracted features of each word are fed into a
# MAGIC forward LSTM network and a backward LSTM network. The
# MAGIC output of each network at each time step is decoded by a linear
# MAGIC layer and a log-softmax layer into log-probabilities for each tag
# MAGIC category. These two vectors are then simply added together to
# MAGIC produce the final output. In the architecture of the proposed framework in the original paper, 50-dimensional pretrained word
# MAGIC embeddings is used for word features, 25-dimension character
# MAGIC embeddings is used for char features, and capitalization features
# MAGIC (allCaps, upperInitial, lowercase, mixedCaps, noinfo) are used
# MAGIC for case features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Blogposts and videos:
# MAGIC 
# MAGIC - [How to Setup Spark NLP for HEALTHCARE on UBUNTU - Video](https://www.youtube.com/watch?v=yKnF-_oz0GE)
# MAGIC 
# MAGIC - [Named Entity Recognition (NER) with BERT in Spark NLP](https://towardsdatascience.com/named-entity-recognition-ner-with-bert-in-spark-nlp-874df20d1d77)
# MAGIC 
# MAGIC - [State of the art Clinical Named Entity Recognition in Spark NLP - Youtube](https://www.youtube.com/watch?v=YM-e4eOiQ34)
# MAGIC 
# MAGIC - [Named Entity Recognition for Healthcare with SparkNLP NerDL and NerCRF](https://medium.com/spark-nlp/named-entity-recognition-for-healthcare-with-sparknlp-nerdl-and-nercrf-a7751b6ad571)
# MAGIC 
# MAGIC - [Named Entity Recognition for Clinical Text](https://medium.com/atlas-research/ner-for-clinical-text-7c73caddd180)

# COMMAND ----------

import os
import json
import string
import numpy as np
import pandas as pd

import nlu
import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.util import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.pretrained import ResourceDownloader

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
# MAGIC # Clinical NER Pipeline (with pretrained models)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clinical NER Models

# COMMAND ----------

# MAGIC %md
# MAGIC - **Clinical NER Models**
# MAGIC 
# MAGIC |index|model|index|model|index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|-----:|:-----|-----:|:-----|
# MAGIC | 1| [jsl_ner_wip_clinical](https://nlp.johnsnowlabs.com/2021/03/31/jsl_ner_wip_clinical_en.html)  | 2| [jsl_ner_wip_greedy_clinical](https://nlp.johnsnowlabs.com/2021/03/31/jsl_ner_wip_greedy_clinical_en.html)  | 3| [jsl_ner_wip_modifier_clinical](https://nlp.johnsnowlabs.com/2021/04/01/jsl_ner_wip_modifier_clinical_en.html)  | 4| [jsl_rd_ner_wip_greedy_clinical](https://nlp.johnsnowlabs.com/2021/04/01/jsl_rd_ner_wip_greedy_clinical_en.html)  |
# MAGIC | 5| [ner_abbreviation_clinical](https://nlp.johnsnowlabs.com/2021/12/30/ner_abbreviation_clinical_en.html)  | 6| [ner_ade_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_clinical_en.html)  | 7| [ner_ade_clinicalbert](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_clinicalbert_en.html)  | 8| [ner_ade_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_healthcare_en.html)  |
# MAGIC | 9| [ner_anatomy](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_en.html)  | 10| [ner_anatomy_coarse](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_en.html)  | 11| [ner_anatomy_coarse_biobert_en](https://nlp.johnsnowlabs.com/2020/11/04/ner_anatomy_coarse_biobert_en.html)  | 12| [ner_anatomy_coarse_en](https://nlp.johnsnowlabs.com/2020/11/04/ner_anatomy_coarse_en.html)  |
# MAGIC | 13| [ner_anatomy_en](https://nlp.johnsnowlabs.com/2020/04/22/ner_anatomy_en.html)  | 14| [ner_aspect_based_sentiment](https://nlp.johnsnowlabs.com/2021/03/31/ner_aspect_based_sentiment_en.html)  | 15| [ner_bacterial_species](https://nlp.johnsnowlabs.com/2021/04/01/ner_bacterial_species_en.html)  | 16| [ner_biomarker](https://nlp.johnsnowlabs.com/2021/11/26/ner_biomarker_en.html)  |
# MAGIC | 17| [ner_biomedical_bc2gm](https://nlp.johnsnowlabs.com/2022/05/11/ner_biomedical_bc2gm_en_2_4.html)  | 18| [ner_bionlp](https://nlp.johnsnowlabs.com/2021/03/31/ner_bionlp_en.html)  | 19| [ner_bionlp_en](https://nlp.johnsnowlabs.com/2020/01/30/ner_bionlp_en.html)  | 20| [ner_cancer_genetics](https://nlp.johnsnowlabs.com/2021/03/31/ner_cancer_genetics_en.html)  |
# MAGIC | 21| [ner_cellular](https://nlp.johnsnowlabs.com/2021/03/31/ner_cellular_en.html)  | 22| [ner_cellular_en](https://nlp.johnsnowlabs.com/2020/04/22/ner_cellular_en.html)  | 23| [ner_chemd_clinical](https://nlp.johnsnowlabs.com/2021/11/04/ner_chemd_clinical_en.html)  | 24| [ner_chemicals](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemicals_en.html)  |
# MAGIC | 25| [ner_chemprot_clinical](https://nlp.johnsnowlabs.com/2020/09/21/ner_chemprot_clinical_en.html)  | 26| [ner_chexpert](https://nlp.johnsnowlabs.com/2021/09/30/ner_chexpert_en.html)  | 27| [ner_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_clinical_en.html)  | 28| [ner_clinical_en](https://nlp.johnsnowlabs.com/2020/01/30/ner_clinical_en.html)  |
# MAGIC | 29| [ner_clinical_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_clinical_large_en.html)  | 30| [ner_clinical_large_en](https://nlp.johnsnowlabs.com/2020/05/23/ner_clinical_large_en.html)  | 31| [ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/06/22/ner_clinical_trials_abstracts_en_3_0.html)  | 32| [ner_covid_trials](https://nlp.johnsnowlabs.com/2021/11/05/ner_covid_trials_en.html)  |
# MAGIC | 33| [ner_crf](https://nlp.johnsnowlabs.com/2020/01/28/ner_crf_en.html)  | 34| [ner_deid_augmented](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_augmented_en.html)  | 35| [ner_deid_enriched](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_enriched_en.html)  | 36| [ner_deid_generic_glove](https://nlp.johnsnowlabs.com/2021/06/06/ner_deid_generic_glove_en.html)  |
# MAGIC | 37| [ner_deid_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_large_en.html)  | 38| [ner_deid_sd](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_en.html)  | 39| [ner_deid_sd_large](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_large_en.html)  | 40| [ner_deid_subentity_augmented](https://nlp.johnsnowlabs.com/2021/09/03/ner_deid_subentity_augmented_en.html)  |
# MAGIC | 41| [ner_deid_subentity_augmented_i2b2](https://nlp.johnsnowlabs.com/2021/11/29/ner_deid_subentity_augmented_i2b2_en.html)  | 42| [ner_deid_subentity_glove](https://nlp.johnsnowlabs.com/2021/06/06/ner_deid_subentity_glove_en.html)  | 43| [ner_deid_synthetic](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_synthetic_en.html)  | 44| [ner_deidentify_dl](https://nlp.johnsnowlabs.com/2021/03/31/ner_deidentify_dl_en.html)  |
# MAGIC | 45| [ner_diseases](https://nlp.johnsnowlabs.com/2021/03/31/ner_diseases_en.html)  | 46| [ner_diseases_en](https://nlp.johnsnowlabs.com/2020/03/25/ner_diseases_en.html)  | 47| [ner_diseases_large](https://nlp.johnsnowlabs.com/2021/04/01/ner_diseases_large_en.html)  | 48| [ner_drugprot_clinical](https://nlp.johnsnowlabs.com/2021/12/20/ner_drugprot_clinical_en.html)  |
# MAGIC | 49| [ner_drugs](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_en.html)  | 50| [ner_drugs_en](https://nlp.johnsnowlabs.com/2020/03/25/ner_drugs_en.html)  | 51| [ner_drugs_greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_greedy_en.html)  | 52| [ner_drugs_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_large_en.html)  |
# MAGIC | 53| [ner_drugs_large_en](https://nlp.johnsnowlabs.com/2021/01/29/ner_drugs_large_en.html)  | 54| [ner_events_admission_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_events_admission_clinical_en.html)  | 55| [ner_events_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_events_clinical_en.html)  | 56| [ner_events_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_healthcare_en.html)  |
# MAGIC | 57| [ner_financial_contract](https://nlp.johnsnowlabs.com/2021/04/01/ner_financial_contract_en.html)  | 58| [ner_genetic_variants](https://nlp.johnsnowlabs.com/2021/06/25/ner_genetic_variants_en.html)  | 59| [ner_healthcare](https://nlp.johnsnowlabs.com/2021/04/21/ner_healthcare_en.html)  | 60| [ner_healthcare_en](https://nlp.johnsnowlabs.com/2020/03/26/ner_healthcare_en.html)  |
# MAGIC | 61| [ner_human_phenotype_gene_clinical](https://nlp.johnsnowlabs.com/2020/09/21/ner_human_phenotype_gene_clinical_en.html)  | 62| [ner_human_phenotype_go_clinical](https://nlp.johnsnowlabs.com/2020/09/21/ner_human_phenotype_go_clinical_en.html)  | 63| [ner_jsl](https://nlp.johnsnowlabs.com/2021/03/31/ner_jsl_en.html)  | 64| [ner_jsl_en](https://nlp.johnsnowlabs.com/2020/04/22/ner_jsl_en.html)  |
# MAGIC | 65| [ner_jsl_enriched](https://nlp.johnsnowlabs.com/2021/10/22/ner_jsl_enriched_en.html)  | 66| [ner_jsl_enriched_en](https://nlp.johnsnowlabs.com/2020/04/22/ner_jsl_enriched_en.html)  | 67| [ner_jsl_greedy](https://nlp.johnsnowlabs.com/2021/06/24/ner_jsl_greedy_en.html)  | 68| [ner_jsl_slim](https://nlp.johnsnowlabs.com/2021/08/13/ner_jsl_slim_en.html)  |
# MAGIC | 69| [ner_living_species](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_en_3_0.html)  | 70| [ner_measurements_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_measurements_clinical_en.html)  | 71| [ner_medmentions_coarse](https://nlp.johnsnowlabs.com/2021/04/01/ner_medmentions_coarse_en.html)  | 72| [ner_nature_nero_clinical](https://nlp.johnsnowlabs.com/2022/02/08/ner_nature_nero_clinical_en.html)  |
# MAGIC | 73| [ner_nihss](https://nlp.johnsnowlabs.com/2021/11/15/ner_nihss_en.html)  | 74| [ner_oncology_anatomy_general_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_anatomy_general_wip_en.html)  | 75| [ner_oncology_anatomy_granular_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_anatomy_granular_wip_en.html)  | 76| [ner_oncology_biomarker_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_biomarker_wip_en.html)  |
# MAGIC | 77| [ner_oncology_demographics_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_demographics_wip_en.html)  | 78| [ner_oncology_diagnosis_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_diagnosis_wip_en.html)  | 79| [ner_oncology_posology_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_posology_wip_en.html)  | 80| [ner_oncology_response_to_treatment_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_response_to_treatment_wip_en.html)  |
# MAGIC | 81| [ner_oncology_test_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_test_wip_en.html)  | 82| [ner_oncology_therapy_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_therapy_wip_en.html)  | 83| [ner_oncology_tnm_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_tnm_wip_en.html)  | 84| [ner_oncology_unspecific_posology_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_unspecific_posology_wip_en.html)  |
# MAGIC | 85| [ner_oncology_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_wip_en.html)  | 86| [ner_pathogen](https://nlp.johnsnowlabs.com/2022/06/28/ner_pathogen_en_3_0.html)  | 87| [ner_posology](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_en.html)  | 88| [ner_posology_en](https://nlp.johnsnowlabs.com/2020/04/15/ner_posology_en.html)  |
# MAGIC | 89| [ner_posology_experimental](https://nlp.johnsnowlabs.com/2021/09/01/ner_posology_experimental_en.html)  | 90| [ner_posology_greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_greedy_en.html)  | 91| [ner_posology_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_healthcare_en.html)  | 92| [ner_posology_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_large_en.html)  |
# MAGIC | 93| [ner_posology_small](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_small_en.html)  | 94| [ner_radiology](https://nlp.johnsnowlabs.com/2021/03/31/ner_radiology_en.html)  | 95| [ner_radiology_wip_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_radiology_wip_clinical_en.html)  | 96| [ner_risk_factors](https://nlp.johnsnowlabs.com/2021/03/31/ner_risk_factors_en.html)  |
# MAGIC | 97| [ner_risk_factors_en](https://nlp.johnsnowlabs.com/2020/04/22/ner_risk_factors_en.html)  | 98| [ner_supplement_clinical](https://nlp.johnsnowlabs.com/2022/02/01/ner_supplement_clinical_en.html)  | 99| [nerdl_tumour_demo](https://nlp.johnsnowlabs.com/2021/04/01/nerdl_tumour_demo_en.html)  | 100| [zero_shot_ner_roberta](https://nlp.johnsnowlabs.com/2022/08/29/zero_shot_ner_roberta_en.html)   |
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC - **BioBert NER Models**
# MAGIC 
# MAGIC |index|model|index|model|index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|-----:|:-----|-----:|:-----|
# MAGIC | 1| [jsl_ner_wip_greedy_biobert](https://nlp.johnsnowlabs.com/2021/07/26/jsl_ner_wip_greedy_biobert_en.html)  | 2| [jsl_rd_ner_wip_greedy_biobert](https://nlp.johnsnowlabs.com/2021/07/26/jsl_rd_ner_wip_greedy_biobert_en.html)  | 3| [ner_ade_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_biobert_en.html)  | 4| [ner_anatomy_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_anatomy_biobert_en.html)  |
# MAGIC | 5| [ner_anatomy_coarse_biobert](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_biobert_en.html)  | 6| [ner_bionlp_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_bionlp_biobert_en.html)  | 7| [ner_cellular_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_cellular_biobert_en.html)  | 8| [ner_chemprot_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemprot_biobert_en.html)  |
# MAGIC | 9| [ner_clinical_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_clinical_biobert_en.html)  | 10| [ner_deid_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_biobert_en.html)  | 11| [ner_deid_enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_enriched_biobert_en.html)  | 12| [ner_diseases_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_diseases_biobert_en.html)  |
# MAGIC | 13| [ner_events_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_biobert_en.html)  | 14| [ner_human_phenotype_gene_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_gene_biobert_en.html)  | 15| [ner_human_phenotype_go_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_go_biobert_en.html)  | 16| [ner_jsl_biobert](https://nlp.johnsnowlabs.com/2021/09/05/ner_jsl_biobert_en.html)  |
# MAGIC | 17| [ner_jsl_enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_jsl_enriched_biobert_en.html)  | 18| [ner_jsl_greedy_biobert](https://nlp.johnsnowlabs.com/2021/08/13/ner_jsl_greedy_biobert_en.html)  | 19| [ner_living_species_biobert](https://nlp.johnsnowlabs.com/2022/06/22/ner_living_species_biobert_en_3_0.html)  | 20| [ner_posology_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_biobert_en.html)  |
# MAGIC | 21| [ner_posology_large_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_large_biobert_en.html)  | 22| [ner_risk_factors_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_risk_factors_biobert_en.html)  | 23| []()| 24| []()|
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC - **BertForTokenClassification Clinical NER models**
# MAGIC 
# MAGIC |index|model|index|model|index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|-----:|:-----|-----:|:-----|
# MAGIC | 1| [bert_token_classifier_ade_tweet_binary](https://nlp.johnsnowlabs.com/2022/07/29/bert_token_classifier_ade_tweet_binary_en_3_0.html)  | 2| [bert_token_classifier_drug_development_trials](https://nlp.johnsnowlabs.com/2022/03/22/bert_token_classifier_drug_development_trials_en_3_0.html)  | 3| [bert_token_classifier_ner_ade](https://nlp.johnsnowlabs.com/2021/09/30/bert_token_classifier_ner_ade_en.html)  | 4| [bert_token_classifier_ner_ade_binary](https://nlp.johnsnowlabs.com/2022/07/27/bert_token_classifier_ner_ade_binary_en_3_0.html)  |
# MAGIC | 5| [bert_token_classifier_ner_anatem](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_anatem_en_3_0.html)  | 6| [bert_token_classifier_ner_anatomy](https://nlp.johnsnowlabs.com/2021/09/30/bert_token_classifier_ner_anatomy_en.html)  | 7| [bert_token_classifier_ner_bacteria](https://nlp.johnsnowlabs.com/2021/09/30/bert_token_classifier_ner_bacteria_en.html)  | 8| [bert_token_classifier_ner_bc2gm_gene](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc2gm_gene_en_3_0.html)  |
# MAGIC | 9| [bert_token_classifier_ner_bc4chemd_chemicals](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc4chemd_chemicals_en_3_0.html)  | 10| [bert_token_classifier_ner_bc5cdr_chemicals](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc5cdr_chemicals_en_3_0.html)  | 11| [bert_token_classifier_ner_bc5cdr_disease](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc5cdr_disease_en_3_0.html)  | 12| [bert_token_classifier_ner_bionlp](https://nlp.johnsnowlabs.com/2021/11/03/bert_token_classifier_ner_bionlp_en.html)  |
# MAGIC | 13| [bert_token_classifier_ner_cellular](https://nlp.johnsnowlabs.com/2021/11/03/bert_token_classifier_ner_cellular_en.html)  | 14| [bert_token_classifier_ner_chemicals](https://nlp.johnsnowlabs.com/2021/10/19/bert_token_classifier_ner_chemicals_en.html)  | 15| [bert_token_classifier_ner_chemprot](https://nlp.johnsnowlabs.com/2021/10/19/bert_token_classifier_ner_chemprot_en.html)  | 16| [bert_token_classifier_ner_clinical](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_clinical_en.html)  |
# MAGIC | 17| [bert_token_classifier_ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/06/29/bert_token_classifier_ner_clinical_trials_abstracts_en_3_0.html)  | 18| [bert_token_classifier_ner_deid](https://nlp.johnsnowlabs.com/2021/09/13/bert_token_classifier_ner_deid_en.html)  | 19| [bert_token_classifier_ner_drugs](https://nlp.johnsnowlabs.com/2021/09/20/bert_token_classifier_ner_drugs_en.html)  | 20| [bert_token_classifier_ner_jnlpba_cellular](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_jnlpba_cellular_en_3_0.html)  |
# MAGIC | 21| [bert_token_classifier_ner_jsl](https://nlp.johnsnowlabs.com/2021/09/16/bert_token_classifier_ner_jsl_en.html)  | 22| [bert_token_classifier_ner_jsl_slim](https://nlp.johnsnowlabs.com/2021/09/24/bert_token_classifier_ner_jsl_slim_en.html)  | 23| [bert_token_classifier_ner_linnaeus_species](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_linnaeus_species_en_3_0.html)  | 24| [bert_token_classifier_ner_living_species](https://nlp.johnsnowlabs.com/2022/06/26/bert_token_classifier_ner_living_species_en_3_0.html)  |
# MAGIC | 25| [bert_token_classifier_ner_ncbi_disease](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_ncbi_disease_en_3_0.html)  | 26| [bert_token_classifier_ner_pathogen](https://nlp.johnsnowlabs.com/2022/07/28/bert_token_classifier_ner_pathogen_en_3_0.html)  | 27| [bert_token_classifier_ner_species](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_species_en_3_0.html)  | 28| [bert_token_classifier_ner_supplement](https://nlp.johnsnowlabs.com/2022/02/09/bert_token_classifier_ner_supplement_en.html)  |
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **You can find all these models and more [NLP Models Hub](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition&edition=Spark+NLP+for+Healthcare)**

# COMMAND ----------

# MAGIC %md
# MAGIC **Also, you can print the list of clinical pretrained models/pipelines and annotators with one-line code:**

# COMMAND ----------

from sparknlp_jsl.pretrained import InternalResourceDownloader

# pipelines = InternalResourceDownloader.returnPrivatePipelines()
ner_models = InternalResourceDownloader.returnPrivateModels("MedicalNerModel")
ner_models

# COMMAND ----------

# Annotator that transforms a text column from dataframe into an Annotation ready for NLP
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
        
sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
 
# Tokenizer splits words in a relevant format for NLP
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

# Clinical word embeddings trained on PubMED dataset
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
    .setInputCols(["sentence","token"])\
    .setOutputCol("embeddings")

# NER model trained on i2b2 (sampled from MIMIC) dataset
clinical_ner = MedicalNerModel.pretrained("ner_clinical_large","en","clinical/models")\
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner")\
    .setLabelCasing("upper") #decide if we want to return the tags in upper or lower case 

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(
    stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        word_embeddings,
        clinical_ner,
        ner_converter
        ])


empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

# COMMAND ----------

#checking the stages in pipeline
model.stages

# COMMAND ----------

#getting the classes in pretrained model
clinical_ner.getClasses()

# COMMAND ----------

#extracting the embedded default param values
clinical_ner.extractParamMap()

# COMMAND ----------

#checking the embeddings
clinical_ner.getStorageRef()

# COMMAND ----------

from sparknlp_jsl.compatibility import Compatibility 
import pandas as pd

compatibility = Compatibility(spark)

models = compatibility.findVersion('ner') 

models_df = pd.DataFrame([dict(x) for x in list(models)])
models_df

# COMMAND ----------

#downloading the sample dataset
! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/mt_samples_10.csv

dbutils.fs.cp("file:/databricks/driver/mt_samples_10.csv", "dbfs:/") 

# COMMAND ----------

mt_samples_df = spark.createDataFrame(pd.read_csv("mt_samples_10.csv", sep=',', index_col=["index"]).reset_index())

# COMMAND ----------

mt_samples_df.printSchema()

# COMMAND ----------

mt_samples_df.show()

# COMMAND ----------

print(mt_samples_df.limit(1).collect()[0]['text'])

# COMMAND ----------

result = model.transform(mt_samples_df).cache()

# COMMAND ----------

result.show()

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC result.select('token.result','ner.result').show(truncate=80)

# COMMAND ----------

result_df = result.select(F.explode(F.arrays_zip(result.token.result,
                                                 result.ner.result, 
                                                 result.ner.metadata)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("ner_label"),
                          F.expr("cols['2']['confidence']").alias("confidence"))

result_df.show(50, truncate=100)

# COMMAND ----------

result_df.select("token", "ner_label")\
         .groupBy('ner_label').count()\
         .orderBy('count', ascending=False)\
         .show(truncate=False)

# COMMAND ----------

result.select('ner_chunk').take(1)

# COMMAND ----------

result.select(F.explode(F.arrays_zip(result.ner_chunk.result, 
                                     result.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"),
              F.expr("cols['1']['confidence']").alias("confidence")).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### with LightPipelines

# COMMAND ----------

# fullAnnotate in LightPipeline

text = '''
A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, and associated with an acute hepatitis, presented with a one-week history of polyuria, poor appetite, and vomiting. 
She was on metformin, glipizide, and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG. She had been on dapagliflozin for six months at the time of presentation. 
Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness, guarding, or rigidity. Pertinent laboratory findings on admission were: serum glucose 111 mg/dl,  creatinine 0.4 mg/dL, triglycerides 508 mg/dL, total cholesterol 122 mg/dL, and venous pH 7.27. 
'''

print (text)

light_model = LightPipeline(model)

light_result = light_model.fullAnnotate(text)


chunks = []
entities = []
sentence= []
begin = []
end = []

for n in light_result[0]['ner_chunk']:
        
    begin.append(n.begin)
    end.append(n.end)
    chunks.append(n.result)
    entities.append(n.metadata['entity']) 
    sentence.append(n.metadata['sentence'])
    
    

df_clinical = pd.DataFrame({'chunks':chunks, 
                            'begin': begin, 
                            'end':end, 
                            'sentence_id':sentence, 
                            'entities':entities})

df_clinical.head(20)

# COMMAND ----------

light_result[0]

# COMMAND ----------

# MAGIC %md
# MAGIC **Prettifying the results with NLU**
# MAGIC 
# MAGIC John Snow Labs' NLU is a Python library for applying state-of-the-art text mining, directly on any dataframe, with a single line of code. We will use some functionalities of NLU to prettify our results.

# COMMAND ----------

nlu.to_pretty_df(model, text, positions=True, output_level='chunk').columns

# COMMAND ----------

#For a given pipeline output level is automatically set to the last anntator's output level by default.
#This can be changed by defining the value of "output_level" parameter.
#Values can be set as; token, sentence, document, chunk and relation .

cols = [
     'entities_ner_chunk',
     'entities_ner_chunk_begin',
     'entities_ner_chunk_end',
     'entities_ner_chunk_origin_sentence',
     'entities_ner_chunk_class',
]
df_clinical = nlu.to_pretty_df(model, text, positions=True, output_level='chunk')[cols]
df_clinical.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NER Visualizer
# MAGIC 
# MAGIC For saving the visualization result as html, provide `save_path` parameter in the display function.

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', return_html=True)

# Change color of an entity label

#visualiser.set_label_colors({'PROBLEM':'#008080', 'TEST':'#800080', 'TREATMENT':'#808080'})
#visualiser.display(light_result[0], label_col='ner_chunk')

# Set label filter

# visualiser.display(light_result, label_col='ner_chunk', document_col='document',
                   #labels=['PROBLEM','TEST'])
  
displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER JSL Model
# MAGIC 
# MAGIC Let's show an example of `ner_jsl` model that has about 80 labels by changing just only the model name.

# COMMAND ----------

# MAGIC %md
# MAGIC **Entities**
# MAGIC 
# MAGIC | | | | | |
# MAGIC |-|-|-|-|-|
# MAGIC |Injury_or_Poisoning|Direction|Test|Admission_Discharge|Death_Entity|
# MAGIC |Relationship_Status|Duration|Respiration|Hyperlipidemia|Birth_Entity|
# MAGIC |Age|Labour_Delivery|Family_History_Header|BMI|Temperature|
# MAGIC |Alcohol|Kidney_Disease|Oncological|Medical_History_Header|Cerebrovascular_Disease|
# MAGIC |Oxygen_Therapy|O2_Saturation|Psychological_Condition|Heart_Disease|Employment|
# MAGIC |Obesity|Disease_Syndrome_Disorder|Pregnancy|ImagingFindings|Procedure|
# MAGIC |Medical_Device|Race_Ethnicity|Section_Header|Symptom|Treatment|
# MAGIC |Substance|Route|Drug_Ingredient|Blood_Pressure|Diet|
# MAGIC |External_body_part_or_region|LDL|VS_Finding|Allergen|EKG_Findings|
# MAGIC |Imaging_Technique|Triglycerides|RelativeTime|Gender|Pulse|
# MAGIC |Social_History_Header|Substance_Quantity|Diabetes|Modifier|Internal_organ_or_component|
# MAGIC |Clinical_Dept|Form|Drug_BrandName|Strength|Fetus_NewBorn|
# MAGIC |RelativeDate|Height|Test_Result|Sexually_Active_or_Sexual_Orientation|Frequency|
# MAGIC |Time|Weight|Vaccine|Vital_Signs_Header|Communicable_Disease|
# MAGIC |Dosage|Overweight|Hypertension|HDL|Total_Cholesterol|
# MAGIC |Smoking|Date||||

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") 

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("jsl_ner")
    
jsl_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "jsl_ner"]) \
    .setOutputCol("jsl_ner_chunk")

jsl_ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    jsl_ner,
    jsl_ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

jsl_ner_model = jsl_ner_pipeline.fit(empty_data)


# COMMAND ----------

jsl_light_model = LightPipeline(jsl_ner_model)

jsl_light_result = jsl_light_model.fullAnnotate(text)

# COMMAND ----------

cols = [
     'entities_jsl_ner_chunk',
     'entities_jsl_ner_chunk_begin',
     'entities_jsl_ner_chunk_end',
     'entities_jsl_ner_chunk_origin_sentence',
     'entities_jsl_ner_chunk_class',
]
jsl_df = nlu.to_pretty_df(jsl_ner_model, text,positions=True, output_level='chunk')[cols]
jsl_df.head(20)

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

ner_vis = visualiser.display(jsl_light_result[0], label_col='jsl_ner_chunk', document_col='document', return_html=True)

displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Posology NER

# COMMAND ----------

# MAGIC %md
# MAGIC **Entities**
# MAGIC 
# MAGIC - DOSAGE
# MAGIC - DRUG
# MAGIC - DURATION
# MAGIC - FORM
# MAGIC - FREQUENCY
# MAGIC - ROUTE
# MAGIC - STRENGTH

# COMMAND ----------

# NER model trained on i2b2 (sampled from MIMIC) dataset
posology_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

posology_ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

# greedy model
posology_ner_greedy = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_greedy")

ner_converter_greedy = NerConverter()\
    .setInputCols(["sentence","token","ner_greedy"])\
    .setOutputCol("ner_chunk_greedy")

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    posology_ner,
    posology_ner_converter,
    posology_ner_greedy,
    ner_converter_greedy])

empty_data = spark.createDataFrame([[""]]).toDF("text")

posology_model = nlpPipeline.fit(empty_data)


# COMMAND ----------

posology_ner.getClasses()

# COMMAND ----------

posology_result = posology_model.transform(mt_samples_df).cache()

# COMMAND ----------

posology_result.show(10)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# This will return a new DF with all the columns + id
posology_result = posology_result.withColumn("id", monotonically_increasing_id())

posology_result.show(3)

# COMMAND ----------

posology_result.select('token.result','ner.result').show(truncate=100)

# COMMAND ----------

posology_result.select('token.result','ner.result').show(5, truncate=80)

# COMMAND ----------

posology_result.select('id',F.explode(F.arrays_zip(posology_result.ner_chunk.result, 
                                                   posology_result.ner_chunk.begin, 
                                                   posology_result.ner_chunk.end, 
                                             posology_result.ner_chunk.metadata)).alias("cols")) \
              .select('id', F.expr("cols['3']['sentence']").alias("sentence_id"),
                            F.expr("cols['0']").alias("chunk"),
                            F.expr("cols['1']").alias("begin"),
                            F.expr("cols['2']").alias("end"),
                            F.expr("cols['3']['entity']").alias("ner_label"),
                            F.expr("cols['3']['confidence']").alias("confidence"))\
              .filter("ner_label!='O'")\
              .show(truncate=False)

# COMMAND ----------

posology_result.select('id',F.explode(F.arrays_zip(posology_result.ner_chunk_greedy.result, 
                                                   posology_result.ner_chunk_greedy.begin, 
                                                   posology_result.ner_chunk_greedy.end, 
                                                   posology_result.ner_chunk_greedy.metadata)).alias("cols")) \
              .select('id', F.expr("cols['3']['sentence']").alias("sentence_id"),
                            F.expr("cols['0']").alias("chunk"),
                            F.expr("cols['1']").alias("begin"),
                            F.expr("cols['2']").alias("end"),
                            F.expr("cols['3']['entity']").alias("ner_label"),
                            F.expr("cols['3']['confidence']").alias("confidence"))\
              .filter("ner_label!='O'")\
              .show(truncate=False)

# COMMAND ----------

posology_result.select('ner_chunk').take(2)[1][0][0].result

# COMMAND ----------

posology_result.select('ner_chunk').take(2)[1][0][0].result

# COMMAND ----------

posology_result.select('ner_chunk').take(2)[1][0][0].metadata

# COMMAND ----------

posology_light_model = LightPipeline(posology_model)

text ='The patient was prescribed 1 capsule of Advil for 5 days . He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely for 3 months .'

posology_light_result = posology_light_model.annotate(text)

list(zip(posology_light_result['token'], posology_light_result['ner']))

# COMMAND ----------

list(zip(posology_light_result['token'], posology_light_result['ner_greedy']))

# COMMAND ----------

cols = [
     'entities_ner_chunk',
     'entities_ner_chunk_class',
     'entities_ner_chunk_begin',
     'entities_ner_chunk_end',
]
posology_result_df = nlu.to_pretty_df(posology_light_model, text, positions=True, output_level='chunk')[cols]
posology_result_df.head(20)

# COMMAND ----------

cols = [
     'entities_ner_chunk_greedy',
     'entities_ner_chunk_greedy_class',
     'entities_ner_chunk_greedy_begin',
     'entities_ner_chunk_greedy_end',
]
posology_result_greedy_df = nlu.to_pretty_df(posology_light_model, text, positions=True, output_level='chunk')[cols]
posology_result_greedy_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NER Visualization

# COMMAND ----------

posology_light_result

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

posology_light_result = posology_light_model.fullAnnotate(text)

ner_vis = visualiser.display(posology_light_result[0], label_col='ner_chunk', document_col='document', return_html=True)
  
displayHTML(ner_vis)

# COMMAND ----------

# ner_greedy

visualiser_greedy = NerVisualizer()

ner_greedy_vis = visualiser_greedy.display(posology_light_result[0], label_col='ner_chunk_greedy', document_col='document', return_html=True)

displayHTML(ner_greedy_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing a generic NER function

# COMMAND ----------

# MAGIC %md
# MAGIC **Generic NER Function with LightPipeline**

# COMMAND ----------

def get_light_model(embeddings, model_name = 'ner_clinical'):

  documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

  sentenceDetector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

  tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

  word_embeddings = WordEmbeddingsModel.pretrained(embeddings, "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

  loaded_ner_model = MedicalNerModel.pretrained(model_name, "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")

  ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "ner"]) \
      .setOutputCol("ner_chunk")

  nlpPipeline = Pipeline(stages=[
      documentAssembler,
      sentenceDetector,
      tokenizer,
      word_embeddings,
      loaded_ner_model,
      ner_converter])

  model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

  return LightPipeline(model)

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_clinical'

light_model = get_light_model(embeddings, model_name)

text = "I had a headache yesterday and took an Advil."

# COMMAND ----------

cols = [
     'entities_ner_chunk_origin_sentence',
     'entities_ner_chunk_begin',
     'entities_ner_chunk_end',
     'entities_ner_chunk',
     'entities_ner_chunk_class',
]

nlu.to_pretty_df(light_model, text, positions=True, output_level='chunk')[cols]

# COMMAND ----------

text ='''The patient was prescribed 1 capsule of Parol with meals . 
He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . 
It was determined that all SGLT2 inhibitors should be discontinued indefinitely fro 3 months .'''

embeddings = 'embeddings_clinical'

model_name = 'ner_posology'

light_model = get_light_model(embeddings, model_name)

# COMMAND ----------

nlu.to_pretty_df(light_model, text,positions=True, output_level='chunk')[cols]

# COMMAND ----------

# MAGIC %md
# MAGIC ## PHI NER

# COMMAND ----------

# MAGIC %md
# MAGIC **Entities**
# MAGIC - AGE
# MAGIC - CONTACT
# MAGIC - DATE
# MAGIC - ID
# MAGIC - LOCATION
# MAGIC - NAME
# MAGIC - PROFESSION

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_deid_subentity_augmented'

# deidentify_dl
# ner_deid_large
# ner_deid_generic_augmented
# ner_deid_subentity_augmented
# ner_deid_subentity_augmented_i2b2

text = """Miriam BRAY is a 41-year-old female from Vietnam and she was admitted for a right-sided pleural effusion for thoracentesis on Monday by Dr. X. Her Coumadin was placed on hold.
She was instructed to followup with Dr. XYZ in the office to check her INR On August 24, 2007 ."""

light_model = get_light_model(embeddings, model_name)

# COMMAND ----------

nlu.to_pretty_df(light_model,text,positions=True,output_level='chunk')[cols]

# COMMAND ----------

# MAGIC %md
# MAGIC ## BioNLP (Cancer Genetics) NER

# COMMAND ----------

# MAGIC %md
# MAGIC **Entities**
# MAGIC 
# MAGIC | | | |
# MAGIC |-|-|-|
# MAGIC |tissue_structure|Amino_acid|Simple_chemical|
# MAGIC |Organism_substance|Developing_anatomical_structure|Cell|
# MAGIC |Cancer|Cellular_component|Gene_or_gene_product|
# MAGIC |Immaterial_anatomical_entity|Organ|Organism|
# MAGIC |Pathological_formation|Organism_subdivision|Anatomical_system|
# MAGIC |Tissue|||

# COMMAND ----------

mt_samples_df.filter("index == '2'").collect()[0]["text"]

# COMMAND ----------

embeddings = 'embeddings_clinical'

model_name = 'ner_bionlp'

text =  mt_samples_df.filter("index == '2'").collect()[0]["text"]

light_model = get_light_model(embeddings, model_name)


# COMMAND ----------

nlu.to_pretty_df(light_model,text,positions=True,output_level='chunk')[cols].head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER Chunker
# MAGIC We can extract phrases that fits into a known pattern using the NER tags. NerChunker would be quite handy to extract entity groups with neighboring tokens when there is no pretrained NER model to address certain issues. Lets say we want to extract drug and frequency together as a single chunk even if there are some unwanted tokens between them.

# COMMAND ----------

posology_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_chunker = NerChunker()\
    .setInputCols(["sentence","ner"])\
    .setOutputCol("ner_chunk")\
    .setRegexParsers(["<DRUG>.*<FREQUENCY>"])

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    posology_ner,
    ner_chunker])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ner_chunker_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

posology_ner.getClasses()

# COMMAND ----------

light_model = LightPipeline(ner_chunker_model)

text ='The patient was prescribed 1 capsule of Advil for 5 days . He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely fro 3 months .'

light_result = light_model.annotate(text)

list(zip(light_result['token'], light_result['ner']))

# COMMAND ----------

light_result["ner_chunk"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Filterer
# MAGIC ChunkFilterer will allow you to filter out named entities by some conditions or predefined look-up lists, so that you can feed these entities to other annotators like Assertion Status or Entity Resolvers. It can be used with two criteria: isin and regex.

# COMMAND ----------

posology_ner = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")
      
chunk_filterer = ChunkFilterer()\
    .setInputCols("sentence","ner_chunk")\
    .setOutputCol("chunk_filtered")\
    .setCriteria("isin")\
    .setWhiteList(['Advil','metformin', 'insulin lispro'])

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    posology_ner,
    ner_converter,
    chunk_filterer])

empty_data = spark.createDataFrame([[""]]).toDF("text")

chunk_filter_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

light_model = LightPipeline(chunk_filter_model)

text ='The patient was prescribed 1 capsule of Advil for 5 days . He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely fro 3 months .'

light_result = light_model.annotate(text)

light_result.keys()

# COMMAND ----------

light_result['ner_chunk'] 

# COMMAND ----------

light_result["chunk_filtered"]

# COMMAND ----------

ner_model = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols("sentence","token","embeddings")\
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")
    
chunk_filterer = ChunkFilterer()\
    .setInputCols("sentence","ner_chunk")\
    .setOutputCol("chunk_filtered")\
    .setCriteria("isin")\
    .setWhiteList(['severe fever','sore throat'])

#   .setCriteria("regex")\

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    ner_model,
    ner_converter,
    chunk_filterer])

empty_data = spark.createDataFrame([[""]]).toDF("text")

chunk_filter_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

text = 'Patient with severe fever, severe cough, sore throat, stomach pain, and a headache.'

filter_df = spark.createDataFrame([[text]]).toDF("text")

chunk_filter_result = chunk_filter_model.transform(filter_df)

# COMMAND ----------

chunk_filter_result.select('ner_chunk.result','chunk_filtered.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Changing entity labels with `NerConverterInternal()`

# COMMAND ----------

replace_dict = """Drug_BrandName,Drug
Frequency,Drug_Frequency
Dosage,Drug_Dosage
Strength,Drug_Strength
"""
with open('replace_dict.csv', 'w') as f:
    f.write(replace_dict)

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") 

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "jsl_ner"]) \
    .setOutputCol("jsl_ner_chunk")

jsl_ner_converter_internal = NerConverterInternal()\
    .setInputCols(["sentence","token","jsl_ner"])\
    .setOutputCol("replaced_ner_chunk")\
    .setReplaceDictResource("file:/databricks/driver/replace_dict.csv","text", {"delimiter":","})
      
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    jsl_ner,
    jsl_ner_converter,
    jsl_ner_converter_internal
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ner_converter_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

text ='The patient was prescribed 1 capsule of Parol with meals. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely fro 3 months .'

#light_model = LightPipeline(ner_converter_model)

cols = [
     'entities_jsl_ner_chunk_origin_sentence',
     'entities_jsl_ner_chunk_begin',
     'entities_jsl_ner_chunk_end',
     'entities_jsl_ner_chunk',
     'entities_jsl_ner_chunk_class',
     'entities_replaced_ner_chunk_class'
]
nlu.to_pretty_df(ner_converter_model,text,positions=True,output_level='chunk')[cols].head(20)

# COMMAND ----------

jsl_ner_converter_internal = NerConverterInternal()\
    .setInputCols(["sentence","token","jsl_ner"])\
    .setOutputCol("replaced_ner_chunk")\
    .setReplaceLabels({"Drug_BrandName": "Drug",
                       "Frequency": "Drug_Frequency",
                       "Dosage": "Drug_Dosage",
                       "Strength": "Drug_Strength"})
      
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    jsl_ner,
    jsl_ner_converter,
    jsl_ner_converter_internal
    ])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ner_converter_model = nlpPipeline.fit(empty_data)

# COMMAND ----------

text ='The patient was prescribed 1 capsule of Parol with meals . He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely for 3 months .'

#light_model = LightPipeline(ner_converter_model)

cols = [
     'entities_jsl_ner_chunk_origin_sentence',
     'entities_jsl_ner_chunk_begin',
     'entities_jsl_ner_chunk_end',
     'entities_jsl_ner_chunk',
     'entities_jsl_ner_chunk_class',
     'entities_replaced_ner_chunk_class'
]
nlu.to_pretty_df(ner_converter_model,text,positions=True,output_level='chunk')[cols].head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Downloading Pretrained Models
# MAGIC 
# MAGIC - When we use `.pretrained` method, model is downloaded to  a folder named `cache_pretrained` automatically and it is loaded from this folder if you run it again.
# MAGIC 
# MAGIC - In order to download the models manually to any folder, you can use `ResourceDownloader.downloadModelDirectly` from `sparknlp.pretrained` or AWS CLI (steps below). In this case you should use `.load()` method.
# MAGIC 
# MAGIC   - Install AWS CLI to your local computer following the steps [here](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html) for Linux and [here](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-mac.html) for MacOS.
# MAGIC 
# MAGIC   - Then configure your AWS credentials.
# MAGIC 
# MAGIC   - Go to models hub and look for the model you need.
# MAGIC 
# MAGIC   - Select the model you found and you will see the model card that shows all the details about that model.
# MAGIC 
# MAGIC   - Hover the Download button on that page and you will see the download link from the S3 bucket.

# COMMAND ----------

from sparknlp.pretrained import ResourceDownloader

#The first argument is the path to the zip file and the second one is the folder.
ResourceDownloader.downloadModelDirectly("clinical/models/ner_jsl_en_3.1.0_2.4_1624566960534.zip", "cache_pretrained")  

#or you can use the classic AWS CLI 
# !aws s3 cp --region us-east-2 s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_en_3.1.0_2.4_1624566960534.zip .

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training a Clinical NER (NCBI Disease Dataset)

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/NER_NCBIconlltrain.txt
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/NER_NCBIconlltest.txt
  
dbutils.fs.cp("file:/databricks/driver/NER_NCBIconlltest.txt", "dbfs:/")
dbutils.fs.cp("file:/databricks/driver/NER_NCBIconlltrain.txt", "dbfs:/")

# COMMAND ----------

from sparknlp.training import CoNLL

conll_data = CoNLL().readDataset(spark, 'file:/databricks/driver/NER_NCBIconlltrain.txt')

conll_data.show(3)

# COMMAND ----------

conll_data.count()

# COMMAND ----------

from pyspark.sql import functions as F

conll_data.select(F.explode(F.arrays_zip(conll_data.token.result, conll_data.label.result)).alias("cols")) \
          .select(F.expr("cols['0']").alias("token"),
                  F.expr("cols['1']").alias("ground_truth")).groupBy('ground_truth').count().orderBy('count', ascending=False).show(100,truncate=False)

# COMMAND ----------

conll_data.select("label.result").distinct().count()

# COMMAND ----------

'''
As you can see, there are too many `O` labels in the dataset. 
To make it more balanced, we can drop the sentences have only O labels.
(`c>1` means we drop all the sentences that have no valuable labels other than `O`)
'''

'''
conll_data = conll_data.withColumn('unique', F.array_distinct("label.result"))\
                       .withColumn('c', F.size('unique'))\
                       .filter(F.col('c')>1)

conll_data.select(F.explode(F.arrays_zip('token.result','label.result')).alias("cols")) \
          .select(F.expr("cols['0']").alias("token"),
                  F.expr("cols['1']").alias("ground_truth"))\
          .groupBy('ground_truth')\
          .count()\
          .orderBy('count', ascending=False)\
          .show(100,truncate=False)
'''

# COMMAND ----------

# Clinical word embeddings trained on PubMED dataset
clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")


# COMMAND ----------

test_data = CoNLL().readDataset(spark, "file:/databricks/driver/NER_NCBIconlltest.txt")

test_data = clinical_embeddings.transform(test_data)

test_data.write.mode("overwrite").parquet('dbfs/NER_NCBIconlltest.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ### NERDL Graph
# MAGIC 
# MAGIC We will use `TFGraphBuilder` annotator which can be used to create graphs in the model training pipeline. `TFGraphBuilder` inspects the data and creates the proper graph if a suitable version of TensorFlow is available. The graph is stored in the defined folder and loaded by the `MedicalNerApproach` annotator.

# COMMAND ----------

#!pip install -q tensorflow==2.7.0
#!pip install -q tensorflow-addons

# COMMAND ----------

from sparknlp_jsl.annotator import TFGraphBuilder

# COMMAND ----------

# MAGIC %md
# MAGIC Firstly, we will create graph and log folder.

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/ner/ner_logs

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/ner/medical_ner_graphs

# COMMAND ----------

graph_folder_path = "/dbfs/ner/medical_ner_graphs"

ner_graph_builder = TFGraphBuilder()\
    .setModelName("ner_dl")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label")\
    .setGraphFolder(graph_folder_path)\
    .setGraphFile("auto")\
    .setHiddenUnitsNumber(20)\
    .setIsMedical(True) # False -> if you want to use TFGraphBuilder with NerDLApproach

# COMMAND ----------

# MAGIC %md
# MAGIC We created ner log and graph files. 
# MAGIC Now, we will create graph and fit the model.

# COMMAND ----------

# TensorFlow graph file (`.pb` extension) can be produced for NER training externally. 
# If this method is used, graph folder should be added to MedicalNerApproach training 
# model as `.setGraphFolder(graph_folder_path)` .

'''
tf_graph.print_model_params("ner_dl")
tf_graph.build("ner_dl", 
               build_params={"embeddings_dim": 200, 
                             "nchars": 85, 
                             "ntags": 12, 
                             "is_medical": 1},
               model_location="/dbfs/ner/medical_ner_graphs", 
               model_filename="auto")
'''

# COMMAND ----------

# for open source users
'''
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/training/english/dl-ner/nerdl-graph/create_graph.py
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/training/english/dl-ner/nerdl-graph/dataset_encoder.py
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/training/english/dl-ner/nerdl-graph/ner_model.py
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/training/english/dl-ner/nerdl-graph/ner_model_saver.py
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/training/english/dl-ner/nerdl-graph/sentence_grouper.py

!pip -q install tensorflow==1.15.0

import create_graph

ntags = 3 # number of labels
embeddings_dim = 200
nchars =83

create_graph.create_graph(ntags, embeddings_dim, nchars)
'''

# COMMAND ----------

nerTagger = MedicalNerApproach()\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setLabelColumn("label")\
  .setOutputCol("ner")\
  .setMaxEpochs(20)\
  .setBatchSize(64)\
  .setRandomSeed(0)\
  .setVerbose(1)\
  .setValidationSplit(0.2)\
  .setEvaluationLogExtended(True) \
  .setEnableOutputLogs(True)\
  .setIncludeConfidence(True)\
  .setTestDataset("/dbfs/NER_NCBIconlltest.parquet")\
  .setOutputLogsPath('dbfs:/ner/ner_logs')\
  .setUseBestModel(True)\
  .setEarlyStoppingCriterion(0.04)\
  .setEarlyStoppingPatience(3)\
  .setGraphFolder('dbfs:/ner/medical_ner_graphs')
  # .setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch       

ner_pipeline = Pipeline(stages=[
          clinical_embeddings,
          ner_graph_builder,
          nerTagger
 ])

# COMMAND ----------

# 20 epochs 3 min
ner_model = ner_pipeline.fit(conll_data)

# if you get an error for incompatible TF graph, use 4.1 NerDL-Graph.ipynb notebook in public folder to create a graph
# licensed users can also use 17.Graph_builder_for_DL_models.ipynb to create tf graphs easily.

# COMMAND ----------

# MAGIC %md
# MAGIC Licensed users can also use [17.Graph_builder_for_DL_models.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb) to create TF graphs easily.

# COMMAND ----------

# MAGIC %md
# MAGIC `getTrainingClassDistribution()` parameter returns the distribution of labels used when training the NER model.

# COMMAND ----------

ner_model.stages[-1].getTrainingClassDistribution()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the results saved in the log file.

# COMMAND ----------

import os 
log_file= os.listdir("/dbfs/ner/ner_logs")[0]

with open (f"/dbfs/ner/ner_logs/{log_file}") as f:
  print(f.read())

# COMMAND ----------

# MAGIC %md
# MAGIC As you see above, our earlyStopping feature worked, trainining was terminated before 30th epoch.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate your model

# COMMAND ----------

pred_df = ner_model.stages[-1].transform(test_data)

# COMMAND ----------

pred_df.columns

# COMMAND ----------

from sparknlp_jsl.eval import NerDLMetrics
import pyspark.sql.functions as F

evaler = NerDLMetrics(mode="full_chunk")

eval_result = evaler.computeMetricsFromDF(pred_df.select("label","ner"), prediction_col="ner", label_col="label", drop_o = True, case_sensitive = True).cache()

eval_result.withColumn("precision", F.round(eval_result["precision"],4))\
           .withColumn("recall", F.round(eval_result["recall"],4))\
           .withColumn("f1", F.round(eval_result["f1"],4)).show(100)

print(eval_result.selectExpr("avg(f1) as macro").show())
print (eval_result.selectExpr("sum(f1*total) as sumprod","sum(total) as sumtotal").selectExpr("sumprod/sumtotal as micro").show())

# COMMAND ----------

evaler = NerDLMetrics(mode="partial_chunk_per_token")

eval_result = evaler.computeMetricsFromDF(pred_df.select("label","ner"), prediction_col="ner", label_col="label", drop_o = True, case_sensitive = True).cache()

eval_result.withColumn("precision", F.round(eval_result["precision"],4))\
           .withColumn("recall", F.round(eval_result["recall"],4))\
           .withColumn("f1", F.round(eval_result["f1"],4)).show(100)

print(eval_result.selectExpr("avg(f1) as macro").show())
print (eval_result.selectExpr("sum(f1*total) as sumprod","sum(total) as sumtotal").selectExpr("sumprod/sumtotal as micro").show())

# COMMAND ----------

ner_model.stages[-1].write().overwrite().save('dbfs:/databricks/driver/models/custom_Med_NER_20e')

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

loaded_ner_model = MedicalNerModel.load("dbfs:/databricks/driver/models/custom_Med_NER_20e")\
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
        clinical_embeddings,
        loaded_ner_model,
        converter])

empty_data = spark.createDataFrame([['']]).toDF("text")

prediction_model = ner_prediction_pipeline.fit(empty_data)

from sparknlp.base import LightPipeline

light_model = LightPipeline(prediction_model)

# COMMAND ----------

text = "She has a metastatic breast cancer to lung"

def get_preds(text, light_model):

    result = light_model.fullAnnotate(text)[0]

    return [(i.result, i.metadata['entity']) for i in result['ner_span']]

get_preds(text, light_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## BertForTokenClassification NER models

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_clinical", "en", "clinical/models")\
    .setInputCols("token", "sentence")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    tokenClassifier,
    ner_converter
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . 
Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . 
She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . 
Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . 
Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . 
Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . 
The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . 
However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . 
The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . 
The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . 
Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . 
The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . 
It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge ."""

res = model.transform(spark.createDataFrame([[text]]).toDF("text"))

# COMMAND ----------

from pyspark.sql import functions as F

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.ner_chunk.begin, res.ner_chunk.end, res.ner_chunk.metadata)).alias("cols")) \
   .select(F.expr("cols['3']['sentence']").alias("sentence_id"),
           F.expr("cols['0']").alias("chunk"),
           F.expr("cols['2']").alias("end"),
           F.expr("cols['3']['entity']").alias("ner_label"))\
   .filter("ner_label!='O'")\
   .show(truncate=False)

# COMMAND ----------

light_model = LightPipeline(model)

light_result = light_model.fullAnnotate(text)

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', save_path="display_bert_result.html")

# COMMAND ----------

# MAGIC %md
# MAGIC **Training Clinical BertForTokenClassification Model**
# MAGIC 
# MAGIC For training own BertForTokenClassification NER model, you can check [this notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.5.BertForTokenClassification_NER_SparkNLP_with_Transformers.ipynb)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pretrained NER Profiling Pipelines
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

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

clinical_profiling_pipeline = PretrainedPipeline("ner_profiling_clinical", "en", "clinical/models")

# COMMAND ----------

text = '''A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .'''


clinical_result = clinical_profiling_pipeline.fullAnnotate(text)[0]
clinical_result.keys()

# COMMAND ----------

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
# MAGIC ## NER Model Finder Pretrained Pipeline
# MAGIC `ner_model_finder`  pretrained pipeline trained with bert embeddings that can be used to find the most appropriate NER model given the entity name.

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline
finder_pipeline = PretrainedPipeline("ner_model_finder", "en", "clinical/models")

# COMMAND ----------

result = finder_pipeline.fullAnnotate("oncology")[0]
result.keys()

# COMMAND ----------

# MAGIC %md
# MAGIC From the metadata in the `model_names` column, we'll get to the top models to the given 'oncology' entity and oncology related categories.

# COMMAND ----------

df= pd.DataFrame(zip(result["model_names"][0].metadata["all_k_resolutions"].split(":::"), 
                     result["model_names"][0].metadata["all_k_results"].split(":::")), 
                 columns=["category", "top_models"])

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #