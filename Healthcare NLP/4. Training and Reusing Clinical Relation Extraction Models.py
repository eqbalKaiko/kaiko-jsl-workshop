# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

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
# MAGIC ## Pretrained Relation Extraction Models and Pipelines
# MAGIC 
# MAGIC Here are the list of pretrained Relation Extraction models and pipelines:
# MAGIC   
# MAGIC   *  **Relation Extraction Models**
# MAGIC 
# MAGIC |index|model|index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|-----:|:-----|
# MAGIC | 1| [posology_re](https://nlp.johnsnowlabs.com/2020/09/01/posology_re.html)  | 16| [re_oncology_size_wip](https://nlp.johnsnowlabs.com/2022/09/26/re_oncology_size_wip_en.html)  | 31| [redl_date_clinical_biobert](https://nlp.johnsnowlabs.com/2021/06/01/redl_date_clinical_biobert_en.html)  |
# MAGIC | 2| [re_ade_biobert](https://nlp.johnsnowlabs.com/2021/07/16/re_ade_biobert_en.html)  | 17| [re_oncology_temporal_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_temporal_wip_en.html)  | 32| [redl_drug_drug_interaction_biobert](https://nlp.johnsnowlabs.com/2021/07/24/redl_drug_drug_interaction_biobert_en.html)  |
# MAGIC | 3| [re_ade_clinical](https://nlp.johnsnowlabs.com/2021/07/12/re_ade_clinical_en.html)  | 18| [re_oncology_test_result_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_test_result_wip_en.html)  | 33| [redl_drugprot_biobert](https://nlp.johnsnowlabs.com/2022/01/05/redl_drugprot_biobert_en.html)  |
# MAGIC | 4| [re_ade_conversational](https://nlp.johnsnowlabs.com/2022/07/27/re_ade_conversational_en_3_0.html)  | 19| [re_oncology_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_wip_en.html)  | 34| [redl_human_phenotype_gene_biobert](https://nlp.johnsnowlabs.com/2021/07/24/redl_human_phenotype_gene_biobert_en.html)  |
# MAGIC | 5| [re_bodypart_directions](https://nlp.johnsnowlabs.com/2021/01/18/re_bodypart_directions_en.html)  | 20| [re_temporal_events_clinical](https://nlp.johnsnowlabs.com/2020/09/28/re_temporal_events_clinical_en.html)  | 35| [redl_nihss_biobert](https://nlp.johnsnowlabs.com/2021/11/16/redl_nihss_biobert_en.html)  |
# MAGIC | 6| [re_bodypart_problem](https://nlp.johnsnowlabs.com/2021/01/18/re_bodypart_problem_en.html)  | 21| [re_temporal_events_enriched_clinical](https://nlp.johnsnowlabs.com/2020/09/28/re_temporal_events_enriched_clinical_en.html)  | 36| [redl_oncology_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_biobert_wip_en.html)  |
# MAGIC | 7| [re_bodypart_proceduretest](https://nlp.johnsnowlabs.com/2021/01/18/re_bodypart_proceduretest_en.html)  | 22| [re_test_problem_finding](https://nlp.johnsnowlabs.com/2021/04/19/re_test_problem_finding_en.html)  | 37| [redl_oncology_biomarker_result_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_biomarker_result_biobert_wip_en.html)  |
# MAGIC | 8| [re_clinical](https://nlp.johnsnowlabs.com/2020/09/24/re_clinical_en.html)  | 23| [re_test_result_date](https://nlp.johnsnowlabs.com/2021/02/24/re_test_result_date_en.html)  | 38| [redl_oncology_granular_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_granular_biobert_wip_en.html)  |
# MAGIC | 9| [re_date_clinical](https://nlp.johnsnowlabs.com/2021/01/18/re_date_clinical_en.html)  | 24| [re_zeroshot_biobert](https://nlp.johnsnowlabs.com/2022/04/05/re_zeroshot_biobert_en_3_0.html)  | 39| [redl_oncology_location_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_location_biobert_wip_en.html)  |
# MAGIC | 10| [re_drug_drug_interaction_clinical](https://nlp.johnsnowlabs.com/2020/09/03/re_drug_drug_interaction_clinical_en.html)  | 25| [redl_ade_biobert](https://nlp.johnsnowlabs.com/2021/07/12/redl_ade_biobert_en.html)  | 40| [redl_oncology_size_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/28/redl_oncology_size_biobert_wip_en.html)  |
# MAGIC | 11| [re_drugprot_clinical](https://nlp.johnsnowlabs.com/2022/01/05/re_drugprot_clinical_en.html)  | 26| [redl_bodypart_direction_biobert](https://nlp.johnsnowlabs.com/2021/06/01/redl_bodypart_direction_biobert_en.html)  | 41| [redl_oncology_temporal_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_temporal_biobert_wip_en.html)  |
# MAGIC | 12| [re_human_phenotype_gene_clinical](https://nlp.johnsnowlabs.com/2020/09/30/re_human_phenotype_gene_clinical_en.html)  | 27| [redl_bodypart_problem_biobert](https://nlp.johnsnowlabs.com/2021/06/01/redl_bodypart_problem_biobert_en.html)  | 42| [redl_oncology_test_result_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_test_result_biobert_wip_en.html)  |
# MAGIC | 13| [re_oncology_biomarker_result_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_biomarker_result_wip_en.html)  | 28| [redl_bodypart_procedure_test_biobert](https://nlp.johnsnowlabs.com/2021/09/10/redl_bodypart_procedure_test_biobert_en.html)  | 43| [redl_temporal_events_biobert](https://nlp.johnsnowlabs.com/2021/07/24/redl_temporal_events_biobert_en.html)  |
# MAGIC | 14| [re_oncology_granular_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_granular_wip_en.html)  | 29| [redl_chemprot_biobert](https://nlp.johnsnowlabs.com/2021/07/24/redl_chemprot_biobert_en.html)  | |
# MAGIC | 15| [re_oncology_location_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_location_wip_en.html)  | 30| [redl_clinical_biobert](https://nlp.johnsnowlabs.com/2021/07/24/redl_clinical_biobert_en.html)  | |
# MAGIC 
# MAGIC *  **Relation Extraction Pipelines**
# MAGIC 
# MAGIC |index|model|index|model|index|model|
# MAGIC |-----:|:-----|-----:|:-----|-----:|:-----|
# MAGIC | 1| [re_bodypart_directions_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_bodypart_directions_pipeline_en_3_0.html)  | 4| [re_human_phenotype_gene_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_human_phenotype_gene_clinical_pipeline_en_3_0.html)  | 7| [re_test_problem_finding_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_test_problem_finding_pipeline_en_3_0.html)  |
# MAGIC | 2| [re_bodypart_proceduretest_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_bodypart_proceduretest_pipeline_en_3_0.html)  | 5| [re_temporal_events_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_temporal_events_clinical_pipeline_en_3_0.html)  | 8| [re_test_result_date_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_test_result_date_pipeline_en_3_0.html)  |
# MAGIC | 3| [re_date_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_date_clinical_pipeline_en_3_0.html)  | 6| [re_temporal_events_enriched_clinical_pipeline](https://nlp.johnsnowlabs.com/2022/03/31/re_temporal_events_enriched_clinical_pipeline_en_3_0.html)  | |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Posology Releation Extraction
# MAGIC 
# MAGIC This is a demonstration of using SparkNLP for extracting posology relations. The following relatios are supported:
# MAGIC 
# MAGIC DRUG-DOSAGE
# MAGIC DRUG-FREQUENCY
# MAGIC DRUG-ADE (Adversed Drug Events)
# MAGIC DRUG-FORM
# MAGIC DRUG-ROUTE
# MAGIC DRUG-DURATION
# MAGIC DRUG-REASON
# MAGIC DRUG=STRENGTH
# MAGIC 
# MAGIC The model has been validated against the posology dataset described in (Magge, Scotch, & Gonzalez-Hernandez, 2018).
# MAGIC 
# MAGIC | Relation | Recall | Precision | F1 | F1 (Magge, Scotch, & Gonzalez-Hernandez, 2018) |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | DRUG-ADE | 0.66 | 1.00 | **0.80** | 0.76 |
# MAGIC | DRUG-DOSAGE | 0.89 | 1.00 | **0.94** | 0.91 |
# MAGIC | DRUG-DURATION | 0.75 | 1.00 | **0.85** | 0.92 |
# MAGIC | DRUG-FORM | 0.88 | 1.00 | **0.94** | 0.95* |
# MAGIC | DRUG-FREQUENCY | 0.79 | 1.00 | **0.88** | 0.90 |
# MAGIC | DRUG-REASON | 0.60 | 1.00 | **0.75** | 0.70 |
# MAGIC | DRUG-ROUTE | 0.79 | 1.00 | **0.88** | 0.95* |
# MAGIC | DRUG-STRENGTH | 0.95 | 1.00 | **0.98** | 0.97 |
# MAGIC 
# MAGIC 
# MAGIC *Magge, Scotch, Gonzalez-Hernandez (2018) collapsed DRUG-FORM and DRUG-ROUTE into a single relation.

# COMMAND ----------

import functools 
import numpy as np
from scipy import spatial
import pyspark.sql.functions as F
import pyspark.sql.types as T
from sparknlp.base import *

# COMMAND ----------

# MAGIC %md
# MAGIC **Build pipeline using SparNLP pretrained models and the relation extration model optimized for posology**.
# MAGIC  
# MAGIC  The precision of the RE model is controlled by "setMaxSyntacticDistance(4)", which sets the maximum syntactic distance between named entities to 4. A larger value will improve recall at the expense at lower precision. A value of 4 leads to literally perfect precision (i.e. the model doesn't produce any false positives) and reasonably good recall.

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = sparknlp.annotators.Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

ner_tagger = MedicalNerModel()\
    .pretrained("ner_posology", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

reModel = RelationExtractionModel()\
    .pretrained("posology_re")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_chunker,
    dependency_parser,
    reModel
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)


# COMMAND ----------

# MAGIC %md
# MAGIC **Create a light pipeline for annotating free text**

# COMMAND ----------

text = """
The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also 
given 1 unit of Metformin daily.
He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 
12 units of insulin lispro with meals , and metformin 1000 mg two times a day.
"""

lmodel = LightPipeline(model)

results = lmodel.fullAnnotate(text)

# COMMAND ----------

results[0].keys()

# COMMAND ----------

results[0]['ner_chunks']

# COMMAND ----------

results[0]['relations']

# COMMAND ----------

# MAGIC %md
# MAGIC **Show extracted relations**

# COMMAND ----------

for rel in results[0]["relations"]:
    print("{}({}={} - {}={})".format(
        rel.result,
        rel.metadata['entity1'], 
        rel.metadata['chunk1'], 
        rel.metadata['entity2'],
        rel.metadata['chunk2']
    ))

# COMMAND ----------

import pandas as pd

# get relations in a pandas dataframe

def get_relations_df (results, rel_col='relations', chunk_col='ner_chunks'):
    rel_pairs=[]
    chunks = []

    for rel in results[0][rel_col]:
        rel_pairs.append((
            rel.metadata['entity1_begin'],
            rel.metadata['entity1_end'],
            rel.metadata['chunk1'], 
            rel.metadata['entity1'], 
            rel.metadata['entity2_begin'],
            rel.metadata['entity2_end'],
            rel.metadata['chunk2'], 
            rel.metadata['entity2'],
            rel.result, 
            rel.metadata['confidence'],
        ))

    for chunk in results[0][chunk_col]:
        chunks.append((
            chunk.metadata["sentence"],
            chunk.begin,
            chunk.end,
            chunk.result, 
        ))

    rel_df = pd.DataFrame(rel_pairs, columns=['entity1_begin', 'entity1_end', 'chunk1', 'entity1', 'entity2_begin', 'entity2_end', 'chunk2', 'entity2', 'relation', 'confidence'])

    chunks_df = pd.DataFrame(chunks, columns = ["sentence", "begin", "end", "chunk"])
    chunks_df.begin = chunks_df.begin.astype(str)
    chunks_df.end = chunks_df.end.astype(str)

    result_df = pd.merge(rel_df,chunks_df, left_on=["entity1_begin", "entity1_end", "chunk1"], right_on=["begin", "end", "chunk"])[["sentence"] + list(rel_df.columns)]


    return result_df

# COMMAND ----------

print(text, "\n")

rel_df = get_relations_df (results)
rel_df

# COMMAND ----------

text ="""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), 
one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . 
She had close follow-up with endocrinology post discharge .
"""

annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df



# COMMAND ----------

# MAGIC %md
# MAGIC **Visualization of Extracted Relations**

# COMMAND ----------

text = """
The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also 
given 1 unit of Metformin daily.
He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine , 
12 units of insulin lispro with meals , and metformin two times a day.
"""

lmodel = LightPipeline(model)

results = lmodel.fullAnnotate(text)

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer


visualizer = RelationExtractionVisualizer()
vis = visualizer.display(results[0], 'relations', show_relations=True, return_html=True) # default show_relations: True

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ADE Relation Extraction
# MAGIC 
# MAGIC We can extract relations between `DRUG` and adverse reactions `ADE` caused by them. Also we can find if an adverse event is caused by a drug or not.
# MAGIC 
# MAGIC `1`: Shows the adverse event and drug entities are related.
# MAGIC 
# MAGIC `0`: Shows the adverse event and drug entities are not related.

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will first use `ner_ade_clinical` NER model and detect `DRUG` and `ADE` entities. Then we ca find the relations between them by using `re_ade_clinical` model. We can also use the same pipeline elemnets rest of them we created above.

# COMMAND ----------

ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")  

reModel = RelationExtractionModel()\
    .pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(["drug-ade, ade-drug"])\
    .setRelationPairsCaseSensitive(False)     # it will return any "ade-drug" or "ADE-DRUG" relationship. By default it’s set to False
                                              # True, then the pairs of entities in the dataset should match the pairs in setRelationPairs in their specific case (case sensitive). 
                                              # False, meaning that the match of those relation names is case insensitive.


ade_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_chunker,
    dependency_parser,
    reModel
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_model = ade_pipeline.fit(empty_data)

# COMMAND ----------

# MAGIC %md
# MAGIC **Show the classes**

# COMMAND ----------

reModel.getClasses()

# COMMAND ----------

# MAGIC %md
# MAGIC **We will create a LightPipeline for annotation.**

# COMMAND ----------

ade_lmodel = LightPipeline(ade_model)

# COMMAND ----------

text = "I experienced fatigue, muscle cramps, anxiety, agression and sadness after taking Lipitor but no more adverse after passing Zocor."


ade_results = ade_lmodel.fullAnnotate(text)

# COMMAND ----------

ade_results[0].keys()

# COMMAND ----------

ade_results[0]["ner_chunks"]

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets show the `ADE`-`DRUG` relations by using pandas dataframe.**

# COMMAND ----------

print(text)

rel_df = get_relations_df (ade_results)

rel_df


# COMMAND ----------

# MAGIC %md
# MAGIC **Visualization of relations**

# COMMAND ----------

visualizer = RelationExtractionVisualizer()
vis = visualizer.display(ade_results[0], 'relations', show_relations=True, return_html=True) # default show_relations: True

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets set custom labels instead of default ones by using  `.setCustomLabels()` parameter**

# COMMAND ----------

reModel = RelationExtractionModel()\
    .pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(["drug-ade, ade-drug"])\
    .setCustomLabels({"1": "is_related", "0": "not_related"})

ade_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_chunker,
    dependency_parser,
    reModel
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_model = ade_pipeline.fit(empty_data)

# COMMAND ----------

ade_lmodel = LightPipeline(ade_model)

text ="""A 44-year-old man taking naproxen for chronic low back pain and a 20-year-old woman on oxaprozin for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands."""

ade_results = ade_lmodel.fullAnnotate(text)

# COMMAND ----------

print(text)

rel_df = get_relations_df (ade_results)

rel_df


# COMMAND ----------

visualizer = RelationExtractionVisualizer()
vis = visualizer.display(ade_results[0], 'relations', show_relations=True, return_html=True) # default show_relations: True

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clinical RE

# COMMAND ----------

# MAGIC %md
# MAGIC **The set of relations defined in the 2010 i2b2 relation challenge**
# MAGIC 
# MAGIC TrIP: A certain treatment has improved or cured a medical problem (eg, ‘infection resolved with antibiotic course’)
# MAGIC 
# MAGIC TrWP: A patient's medical problem has deteriorated or worsened because of or in spite of a treatment being administered (eg, ‘the tumor was growing despite the drain’)
# MAGIC 
# MAGIC TrCP: A treatment caused a medical problem (eg, ‘penicillin causes a rash’)
# MAGIC 
# MAGIC TrAP: A treatment administered for a medical problem (eg, ‘Dexamphetamine for narcolepsy’)
# MAGIC 
# MAGIC TrNAP: The administration of a treatment was avoided because of a medical problem (eg, ‘Ralafen which is contra-indicated because of ulcers’)
# MAGIC 
# MAGIC TeRP: A test has revealed some medical problem (eg, ‘an echocardiogram revealed a pericardial effusion’)
# MAGIC 
# MAGIC TeCP: A test was performed to investigate a medical problem (eg, ‘chest x-ray done to rule out pneumonia’)
# MAGIC 
# MAGIC PIP: Two problems are related to each other (eg, ‘Azotemia presumed secondary to sepsis’)

# COMMAND ----------

clinical_ner_tagger = MedicalNerModel()\
    .pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    .setRelationPairs(["problem-test", "problem-treatment"]) # we can set the possible relation pairs (if not set, all the relations will be calculated)

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    clinical_ner_tagger,
    ner_chunker,
    dependency_parser,
    clinical_re_Model
])


empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

text ="""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), 
one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . 
She had close follow-up with endocrinology post discharge .
"""

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df[rel_df.relation!="O"]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Clinical Temporal Events RE

# COMMAND ----------

# MAGIC %md
# MAGIC Temporal relations, or temporal links (denoted by the TLINK tag), indicate whether and how two EVENTs, two TIME, or an EVENT and a TIME related to each other in the clinical timeline. There are 3 type of relations here and below are some examples of Relations, with square brackets indicating EVENT and TIME connected by a temporal link:
# MAGIC 
# MAGIC **`BEFORE`**
# MAGIC 
# MAGIC The patient was given stress dose steroids prior to his surgery. ([stress dose steroids] `BEFORE` [his surgery])
# MAGIC 
# MAGIC The patient had an undocumented history of possible atrial fibrillation prior to admission. ([possible atrial fibrillation] `BEFORE` [admission])
# MAGIC 
# MAGIC His nasogastric tube was discontinued on 05-26-98. ([His nasogastric] `BEFORE` [05-26-98])
# MAGIC 
# MAGIC **`AFTER`**
# MAGIC 
# MAGIC Before admission, he had another serious concussion. ([admission] `AFTER` [another serious concussion])
# MAGIC 
# MAGIC On postoperative day No 1, he was started on Percocet. ([Percocet] `AFTER` [postoperative day No 1])
# MAGIC 
# MAGIC 
# MAGIC **`OVERLAP`**
# MAGIC 
# MAGIC She denies any fevers or chills. ([fevers] `OVERLAP` [chills])
# MAGIC 
# MAGIC The patient's serum creatinine on discharge date, 2012-05-06, was 1.9. ([discharge date] `OVERLAP` [2012-05-06])
# MAGIC 
# MAGIC His preoperative workup was completed and included a normal white count ([a normal white count] `OVERLAP` [His preoperative workup])
# MAGIC 
# MAGIC The patient had an undocumented history of possible atrial fibrillation prior to admission. ([possible atrial fibrillation] `OVERLAP` [admission])
# MAGIC 
# MAGIC 
# MAGIC | Relation | Recall | Precision | F1 |
# MAGIC | --- | --- | --- | --- |
# MAGIC | OVERLAP | 0.81 | 0.73 | **0.77** |
# MAGIC | BEFORE | 0.85 | 0.88 | **0.86** |
# MAGIC | AFTER | 0.38 | 0.46 | **0.43** |

# COMMAND ----------

# MAGIC %md
# MAGIC This RE model works with `ner_events_clinical` NER model and expect the following entities as inputs:
# MAGIC 
# MAGIC [`OCCURRENCE`,
# MAGIC  `DATE`,
# MAGIC  `DURATION`,
# MAGIC  `EVIDENTIAL`,
# MAGIC  `TEST`,
# MAGIC  `PROBLEM`,
# MAGIC  `TREATMENT`,
# MAGIC  `CLINICAL_DEPT`,
# MAGIC  `FREQUENCY`,
# MAGIC  `TIME`]

# COMMAND ----------

events_ner_tagger = MedicalNerModel()\
    .pretrained("ner_events_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_temporal_events_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    .setPredictionThreshold(0.9)

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    events_ner_tagger,
    ner_chunker,
    dependency_parser,
    clinical_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

events_ner_tagger.getClasses()

# COMMAND ----------

text ="She had a retinal degeneration after the accident in 2005 May"

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df = rel_df[(rel_df.relation!="O")]

rel_df

# COMMAND ----------

text ="On 9–28-92, the patient will return for chemotherapy and she will follow up with her primary doctor, for PT and Coumadin dosing on Monday."

annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df.confidence = rel_df.confidence.astype(float)

rel_df = rel_df[(rel_df.relation!="O")]

rel_df[(rel_df.relation!="O")&(rel_df.entity1!=rel_df.entity2)]


# COMMAND ----------

# MAGIC %md
# MAGIC **Admission-Discharge Events Exraction**
# MAGIC 
# MAGIC We can also extract `ADMISSION` and `DISCHARGE` entities in addition to `ner_events_clinical` model entities using `ner_events_admission_clinical` NER model and expect the following entities as inputs:
# MAGIC    
# MAGIC [`EVIDENTIAL`,
# MAGIC  `OCCURRENCE`,
# MAGIC  `TREATMENT`,
# MAGIC  `DATE`,
# MAGIC  `ADMISSION`,
# MAGIC  `TIME`,
# MAGIC  `FREQUENCY`,
# MAGIC  `CLINICAL_DEPT`,
# MAGIC  `DURATION`,
# MAGIC  `PROBLEM`,
# MAGIC  `TEST`,
# MAGIC  `DISCHARGE`]

# COMMAND ----------

events_admission_ner_tagger = MedicalNerModel()\
    .pretrained("ner_events_admission_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    events_admission_ner_tagger,
    ner_chunker,
    dependency_parser,
    clinical_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

events_admission_ner_tagger.getClasses()

# COMMAND ----------

text ="""She is admitted to The John Hopkins Hospital on Monday with a history of gestational diabetes mellitus diagnosed.  
She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 
12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. 
"""

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df.confidence = rel_df.confidence.astype(float)

rel_df[(rel_df.relation!="O")]


# COMMAND ----------

# MAGIC %md
# MAGIC **Test-Result-Date Extraction**
# MAGIC 
# MAGIC There is another Relation Extraction model to link clinical tests to test results and dates to clinical entities: `re_test_result_date`

# COMMAND ----------

trd_ner_tagger = MedicalNerModel()\
    .pretrained("jsl_ner_wip_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")       

trd_re_Model = RelationExtractionModel()\
    .pretrained("re_test_result_date", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    trd_ner_tagger,
    ner_chunker,
    dependency_parser,
    trd_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

text = 'Hospitalized with pneumonia on 11 June 2019, confirmed by a positive PCR of any specimen, evidenced by SPO2 </= 93%.'

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df.confidence = rel_df.confidence.astype(float)

rel_df[(rel_df.relation!="O")]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Human Phenotype - Gene RE

# COMMAND ----------

# MAGIC %md
# MAGIC https://github.com/lasigeBioTM/PGR
# MAGIC 
# MAGIC Human phenotype-gene relations are fundamental to fully understand the origin of some phenotypic abnormalities and their associated diseases. Biomedical literature is the most comprehensive source of these relations, however, we need Relation Extraction tools to automatically recognize them. We present the Phenotype-Gene Relations (PGR) model, trained on a silver standard corpus of human phenotype and gene annotations and their relations. 
# MAGIC 
# MAGIC It extracts 2 label: `True` or `False`

# COMMAND ----------

pgr_ner_tagger = MedicalNerModel()\
    .pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

pgr_re_Model = RelationExtractionModel()\
    .pretrained("re_human_phenotype_gene_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setRelationPairs(["hp-gene",'gene-hp'])\
    .setMaxSyntacticDistance(4)\

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    pgr_ner_tagger,
    ner_chunker,
    dependency_parser,
    pgr_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

text = "She has a retinal degeneration, hearing loss and renal failure, short stature, \
Mutations in the SH3PXD2B gene coding for the Tks4 protein are responsible for the autosomal recessive. "

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df = rel_df[(rel_df.relation!=0)]

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drug-Drug Interaction RE

# COMMAND ----------

# MAGIC %md
# MAGIC In clinical application, two or more drugs are often used in combination to achieve conducive results, such as synergistic effect, increasing therapeutic effect and reducing or delaying the occurrence of
# MAGIC drug resistance. However, there is a potential for harmful drug-drug
# MAGIC interactions (DDIs) to occur when two or more drugs are taken at the
# MAGIC same time or at certain interval, which can reduce or invalidate the
# MAGIC efficacy of drugs, and increase toxicity or even cause death. Therefore,
# MAGIC in order to prevent harmful drug-drug interaction (DDI), medical staff
# MAGIC often spend much time in reviewing the relevant drug alert literature
# MAGIC and drug knowledge bases. 
# MAGIC 
# MAGIC **ref**: *Drug-drug interaction extraction via hybrid neural networks on biomedical literature
# MAGIC https://www-sciencedirect-com.ezproxy.leidenuniv.nl:2443/science/article/pii/S1532046420300605*

# COMMAND ----------

ner_tagger = MedicalNerModel()\
    .pretrained("ner_posology", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   

ner_converter = NerConverterInternal() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunk")

ddi_re_model = RelationExtractionModel()\
    .pretrained("re_drug_drug_interaction_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
    .setOutputCol("relations")\
    .setRelationPairs(["drug-drug"])\
    .setMaxSyntacticDistance(4)\

ddi_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_converter,
    dependency_parser,
    ddi_re_model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ddi_model = ddi_pipeline.fit(empty_data)

# COMMAND ----------

text='When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced. \
If additional adrenergic drugs are to be administered by any route, \
they should be used with caution because the pharmacologically predictable sympathetic effects of Metformin may be potentiated'

lmodel = LightPipeline(ddi_model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations, chunk_col="ner_chunk")

rel_df

# COMMAND ----------

annotations[0]['ner_chunk']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drug-Protein Interaction RE

# COMMAND ----------

# MAGIC %md
# MAGIC Detect interactions between chemical compounds/drugs and genes/proteins using Spark NLP’s `RelationExtractionModel()` by classifying whether a specified semantic relation holds between a chemical and gene entities within a sentence or document. <br/>
# MAGIC 
# MAGIC The entity labels used during training were derived from the custom NER model created by our team for the **DrugProt** corpus. These include `CHEMICAL` for chemical compounds/drugs, `GENE` for genes/proteins and `GENE_AND_CHEMICAL` for entity mentions of type `GENE` and of type `CHEMICAL` that overlap (such as enzymes and small peptides). <br/>
# MAGIC 
# MAGIC The relation categories from the **DrugProt** corpus were condensed from 13 categories to 10 categories due to low numbers of examples for certain categories. This merging process involved grouping the `SUBSTRATE_PRODUCT-OF` and `SUBSTRATE` relation categories together and grouping the `AGONIST-ACTIVATOR`, `AGONIST-INHIBITOR` and `AGONIST` relation categories together.
# MAGIC 
# MAGIC **ref**: *DrugProt corpus: Biocreative VII Track 1 - Text mining drug and chemical-protein interactionshttps://zenodo.org/record/5119892*

# COMMAND ----------

drugprot_ner_tagger = MedicalNerModel.pretrained("ner_drugprot_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   

ner_converter = NerConverter()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

drugprot_re_model = RelationExtractionModel()\
        .pretrained("re_drugprot_clinical", "en", 'clinical/models')\
        .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
        .setOutputCol("relations")\
        .setMaxSyntacticDistance(4)\
        .setPredictionThreshold(0.5)\
        .setRelationPairs(["checmical-gene", "chemical-gene_and_chemical", "gene_and_chemical-gene"]) 

pipeline = Pipeline(stages=[
        documenter, 
        sentencer,
        tokenizer,
        words_embedder, 
        drugprot_ner_tagger,
        ner_converter, 
        pos_tagger, 
        dependency_parser, 
        drugprot_re_model])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

text='''Lipid specific activation of the murine P4-ATPase Atp8a1 (ATPase II). The asymmetric transbilayer distribution of phosphatidylserine (PS) in the mammalian plasma membrane and secretory vesicles is maintained, in part, by an ATP-dependent transporter. This aminophospholipid "flippase" selectively transports PS to the cytosolic leaflet of the bilayer and is sensitive to vanadate, Ca(2+), and modification by sulfhydryl reagents. Although the flippase has not been positively identified, a subfamily of P-type ATPases has been proposed to function as transporters of amphipaths, including PS and other phospholipids. A candidate PS flippase ATP8A1 (ATPase II), originally isolated from bovine secretory vesicles, is a member of this subfamily based on sequence homology to the founding member of the subfamily, the yeast protein Drs2, which has been linked to ribosomal assembly, the formation of Golgi-coated vesicles, and the maintenance of PS asymmetry. To determine if ATP8A1 has biochemical characteristics consistent with a PS flippase, a murine homologue of this enzyme was expressed in insect cells and purified. The purified Atp8a1 is inactive in detergent micelles or in micelles containing phosphatidylcholine, phosphatidic acid, or phosphatidylinositol, is minimally activated by phosphatidylglycerol or phosphatidylethanolamine (PE), and is maximally activated by PS. The selectivity for PS is dependent upon multiple elements of the lipid structure. Similar to the plasma membrane PS transporter, Atp8a1 is activated only by the naturally occurring sn-1,2-glycerol isomer of PS and not the sn-2,3-glycerol stereoisomer. Both flippase and Atp8a1 activities are insensitive to the stereochemistry of the serine headgroup. Most modifications of the PS headgroup structure decrease recognition by the plasma membrane PS flippase. Activation of Atp8a1 is also reduced by these modifications; phosphatidylserine-O-methyl ester, lysophosphatidylserine, glycerophosphoserine, and phosphoserine, which are not transported by the plasma membrane flippase, do not activate Atp8a1. Weakly translocated lipids (PE, phosphatidylhydroxypropionate, and phosphatidylhomoserine) are also weak Atp8a1 activators. However, N-methyl-phosphatidylserine, which is transported by the plasma membrane flippase at a rate equivalent to PS, is incapable of activating Atp8a1 activity. These results indicate that the ATPase activity of the secretory granule Atp8a1 is activated by phospholipids binding to a specific site whose properties (PS selectivity, dependence upon glycerol but not serine, stereochemistry, and vanadate sensitivity) are similar to, but distinct from, the properties of the substrate binding site of the plasma membrane flippase.'''

lmodel = LightPipeline(model)

results = lmodel.fullAnnotate(text)

# COMMAND ----------

results[0].keys()

# COMMAND ----------

results[0]["ner_chunks"]

# COMMAND ----------

results[0]["relations"]

# COMMAND ----------

rel_df = get_relations_df (results)

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chemical–Protein Interactions (ChemProt RE)

# COMMAND ----------

# MAGIC %md
# MAGIC Accurately detecting the interactions between chemicals and proteins is a crucial task that plays a key role in precision medicine, drug discovery and basic clinical research. Currently, PubMed contains >28 million articles, and its annual growth rate is more than a million articles each year. A large amount of valuable chemical–protein interactions (CPIs) are hidden in the biomedical literature. There is an increasing interest in CPI extraction from the biomedical literature.
# MAGIC 
# MAGIC Since manually extracting biomedical relations such as protein–protein interactions (PPI) and drug–drug interactions (DDI) is costly and time-consuming, some computational methods have been successfully proposed for automatic biomedical relation extraction.
# MAGIC 
# MAGIC To date, most studies on the biomedical relation extraction have focused on the PPIs and DDIs, but a few attempts have been made to extract CPIs. The BioCreative VI ChemProt shared task released the ChemProt dataset for CPI extraction, which is the first challenge for extracting CPIs.
# MAGIC 
# MAGIC Computational CPI extraction is generally approached as a task of classifying whether a specified semantic relation holds between the chemical and protein entities within a sentence or document. The ChemProt corpus is a manually annotated CPI dataset, which greatly promotes the development of CPI extraction approaches. 
# MAGIC 
# MAGIC ref: https://academic.oup.com/database/article/doi/10.1093/database/baz054/5498050

# COMMAND ----------

# MAGIC %md
# MAGIC | Relation | Recall | Precision | F1 | F1 (Zhang, Yijia, et al., 2019) |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | CPR:3 | 0.47 | 0.59 | **0.52** | 0.594 |
# MAGIC | CPR:4 | 0.72 | 0.81 | **0.77** | 0.718 |
# MAGIC | CPR:5 | 0.43 | 0.88 | **0.58** | 0.657 |
# MAGIC | CPR:6 | 0.59 | 0.89 | **0.71** | 0.725 |
# MAGIC | CPR:9 | 0.62 | 0.84 | **0.71** | 0.501 |
# MAGIC |avg. |  |  | **0.66** | 0.64 |

# COMMAND ----------

# MAGIC %md
# MAGIC Here are the relation types

# COMMAND ----------

# MAGIC %md
# MAGIC | Group | Eval. | CHEMPROT relations belonging to this group |
# MAGIC |-------|-------|--------------------------------------------|
# MAGIC | CPR:1 | N     | PART_OF                                    |
# MAGIC | CPR:2  	| N     	| REGULATOR\|DIRECT REGULATOR\|INDIRECT REGULATOR   	|
# MAGIC | CPR:3  	| Y     	| UPREGULATOR ACTIVATOR INDIRECT UPREGULATOR        	|
# MAGIC | CPR:4  	| Y     	| DOWNREGULATORIINHIBITORIINDIRECTDOWNREGULATOR     	|
# MAGIC | CPR:5  	| Y     	| AGONISTIAGONIST-ACTIVATOR\|AGONIST-INHIBITOR      	|
# MAGIC | CPR:6  	| Y     	| ANTAGONIST                                        	|
# MAGIC | CPR:7  	| N     	| MODULATORIMODULATOR-ACTIVATORIMODULATOR-INHIBITOR 	|
# MAGIC | CPR:8  	| N     	| COFACTOR                                          	|
# MAGIC | CPR:9  	| Y     	| SUBSTRATEIPRODUCT OF\|SUBSTRATE PRODUCT OF        	|
# MAGIC | CPR:10 	| N     	| NOT                                               	|

# COMMAND ----------

# MAGIC %md
# MAGIC ChemProt RE works well with `ner_chemprot_clinical` find relationships between the following entities
# MAGIC 
# MAGIC `CHEMICAL`: Chemical entity mention type; 
# MAGIC 
# MAGIC `GENE-Y`: gene/protein mention type that can be normalized or associated to a biological database identifier; 
# MAGIC 
# MAGIC `GENE-N`: gene/protein mention type that cannot be normalized to a database identifier.

# COMMAND ----------

ner_tagger = MedicalNerModel()\
    .pretrained("ner_chemprot_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   

ner_converter = NerConverterInternal() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunk")

chemprot_re_model = RelationExtractionModel()\
    .pretrained("re_chemprot_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\

chemprot_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_converter,
    dependency_parser,
    chemprot_re_model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

chemprot_model = chemprot_pipeline.fit(empty_data)

# COMMAND ----------

text='''
In this study, we examined the effects of mitiglinide on various cloned K(ATP) channels (Kir6.2/SUR1, Kir6.2/SUR2A, and Kir6.2/SUR2B) reconstituted in COS-1 cells, and compared them to another meglitinide-related compound, nateglinide. Patch-clamp analysis using inside-out recording configuration showed that mitiglinide inhibits the Kir6.2/SUR1 channel currents in a dose-dependent manner (IC50 value, 100 nM) but does not significantly inhibit either Kir6.2/SUR2A or Kir6.2/SUR2B channel currents even at high doses (more than 10 microM). Nateglinide inhibits Kir6.2/SUR1 and Kir6.2/SUR2B channels at 100 nM, and inhibits Kir6.2/SUR2A channels at high concentrations (1 microM). Binding experiments on mitiglinide, nateglinide, and repaglinide to SUR1 expressed in COS-1 cells revealed that they inhibit the binding of [3H]glibenclamide to SUR1 (IC50 values: mitiglinide, 280 nM; nateglinide, 8 microM; repaglinide, 1.6 microM), suggesting that they all share a glibenclamide binding site. The insulin responses to glucose, mitiglinide, tolbutamide, and glibenclamide in MIN6 cells after chronic mitiglinide, nateglinide, or repaglinide treatment were comparable to those after chronic tolbutamide and glibenclamide treatment. These results indicate that, similar to the sulfonylureas, mitiglinide is highly specific to the Kir6.2/SUR1 complex, i.e., the pancreatic beta-cell K(ATP) channel, and suggest that mitiglinide may be a clinically useful anti-diabetic drug.
'''
lmodel = LightPipeline(chemprot_model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations, chunk_col="ner_chunk")

rel_df[rel_df.entity1!=rel_df.entity2]

# COMMAND ----------

# MAGIC %md
# MAGIC # Train a Custom Relation Extraction Model

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/i2b2_clinical_rel_dataset.csv

dbutils.fs.cp("file:/databricks/driver/i2b2_clinical_rel_dataset.csv", "dbfs:/")

# COMMAND ----------

data = spark.read.option("header","true").format("csv").load("/i2b2_clinical_rel_dataset.csv")

data = data.select( 'sentence','firstCharEnt1','firstCharEnt2','lastCharEnt1','lastCharEnt2', "chunk1", "chunk2", "label1", "label2",'rel','dataset')

data.show(10)

# you only need these columns>> 'sentence','firstCharEnt1','firstCharEnt2','lastCharEnt1','lastCharEnt2', "chunk1", "chunk2", "label1", "label2",'rel'
# ('dataset' column is optional)

# COMMAND ----------

data.groupby('dataset').count().show()

# COMMAND ----------

data.groupby('rel').count().show()

# COMMAND ----------

#Annotation structure
annotationType = T.StructType([
            T.StructField('annotatorType', T.StringType(), False),
            T.StructField('begin', T.IntegerType(), False),
            T.StructField('end', T.IntegerType(), False),
            T.StructField('result', T.StringType(), False),
            T.StructField('metadata', T.MapType(T.StringType(), T.StringType()), False),
            T.StructField('embeddings', T.ArrayType(T.FloatType()), False)
        ])

#UDF function to convert train data to names entitities

@F.udf(T.ArrayType(annotationType))
def createTrainAnnotations(begin1, end1, begin2, end2, chunk1, chunk2, label1, label2):
    
    entity1 = sparknlp.annotation.Annotation("chunk", begin1, end1, chunk1, {'entity': label1.upper(), 'sentence': '0'}, [])
    entity2 = sparknlp.annotation.Annotation("chunk", begin2, end2, chunk2, {'entity': label2.upper(), 'sentence': '0'}, [])    
        
    entity1.annotatorType = "chunk"
    entity2.annotatorType = "chunk"

    return [entity1, entity2]    

#list of valid relations
rels = ["TrIP", "TrAP", "TeCP", "TrNAP", "TrCP", "PIP", "TrWP", "TeRP"]

#a query to select list of valid relations
valid_rel_query = "(" + " OR ".join(["rel = '{}'".format(rel) for rel in rels]) + ")"

data = data\
  .withColumn("begin1i", F.expr("cast(firstCharEnt1 AS Int)"))\
  .withColumn("end1i", F.expr("cast(lastCharEnt1 AS Int)"))\
  .withColumn("begin2i", F.expr("cast(firstCharEnt2 AS Int)"))\
  .withColumn("end2i", F.expr("cast(lastCharEnt2 AS Int)"))\
  .where("begin1i IS NOT NULL")\
  .where("end1i IS NOT NULL")\
  .where("begin2i IS NOT NULL")\
  .where("end2i IS NOT NULL")\
  .where(valid_rel_query)\
  .withColumn(
      "train_ner_chunks", 
      createTrainAnnotations(
          "begin1i", "end1i", "begin2i", "end2i", "chunk1", "chunk2", "label1", "label2"
      ).alias("train_ner_chunks", metadata={'annotatorType': "chunk"}))
    

train_data = data.where("dataset='train'")
test_data = data.where("dataset='test'")

# COMMAND ----------

#for training with existing graph 
"""!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/generic_classifier_graph/RE_in1200D_out20.pb -P /databricks/driver/re_graph/"""

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/re_graphs

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/re_logs

# COMMAND ----------

# MAGIC %md
# MAGIC Graph and log folder has been created. 
# MAGIC Now, we will build graph and train the model <br/>

# COMMAND ----------

# TensorFlow graph file (`.pb` extension) can be produced for RE training externally. 
# **WARNING:** For training a Relation Extraction model with custom graph, please use TensorFlow version 2.3
"""
import tensorflow
from sparknlp_jsl.training import tf_graph

tf_graph.print_model_params("relation_extraction")

tf_graph.build("relation_extraction",
               build_params={"input_dim": 1200, 
                             "output_dim": 20, 
                             'batch_norm':1, 
                             "hidden_layers": [300, 200],
                             "hidden_act": "relu", 
                             'hidden_act_l2':1}, 
               model_location="/dbfs/re_graphs", 
               model_filename="rel_e.in1200.out20.pb")"""

# COMMAND ----------

graph_folder= '/dbfs/re_graphs'

re_graph_builder = TFGraphBuilder()\
    .setModelName("relation_extraction")\
    .setInputCols(["embeddings", "pos_tags", "train_ner_chunks", "dependencies"]) \
    .setLabelColumn("rel")\
    .setGraphFolder(graph_folder)\
    .setGraphFile("re_graph.pb")\
    .setHiddenLayers([300, 200])\
    .setHiddenAct("relu")\
    .setHiddenActL2(True)\
    .setHiddenWeightsL2(False)\
    .setBatchNorm(False)

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("sentence")\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")\

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")
    
dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

# set training params and upload model graph (see ../Healthcare/8.Generic_Classifier.ipynb)
reApproach = RelationExtractionApproach()\
    .setInputCols(["embeddings", "pos_tags", "train_ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setLabelColumn("rel")\
    .setEpochsNumber(100)\
    .setBatchSize(200)\
    .setDropout(0.5)\
    .setLearningRate(0.001)\
    .setModelFile(f"{graph_folder}/re_graph.pb")\
    .setFixImbalance(True)\
    .setFromEntity("begin1i", "end1i", "label1")\
    .setToEntity("begin2i", "end2i", "label2")\
    .setOutputLogsPath('dbfs:/re_logs')

finisher = Finisher()\
    .setInputCols(["relations"])\
    .setOutputCols(["relations_out"])\
    .setCleanAnnotations(False)\
    .setValueSplitSymbol(",")\
    .setAnnotationSplitSymbol(",")\
    .setOutputAsArray(False)

train_pipeline = Pipeline(stages=[
    documenter, tokenizer, words_embedder, pos_tagger, 
    dependency_parser, re_graph_builder, reApproach, finisher
])

# COMMAND ----------

rel_model = train_pipeline.fit(train_data)

# COMMAND ----------

rel_model.stages

# COMMAND ----------

rel_model.stages[-2]

# COMMAND ----------

rel_model.stages[-2].write().overwrite().save('dbfs:/databricks/driver/models/custom_RE_model')

# COMMAND ----------

result = rel_model.transform(test_data)

# COMMAND ----------

recall = result\
  .groupBy("rel")\
  .agg(F.avg(F.expr("IF(rel = relations_out, 1, 0)")).alias("recall"))\
  .select(
      F.col("rel").alias("relation"), 
      F.format_number("recall", 2).alias("recall"))\
  .show()

performance  = result\
  .where("relations_out <> ''")\
  .groupBy("relations_out")\
  .agg(F.avg(F.expr("IF(rel = relations_out, 1, 0)")).alias("precision"))\
  .select(
      F.col("relations_out").alias("relation"), 
      F.format_number("precision", 2).alias("precision"))\
  .show()

# COMMAND ----------

result_df = result.select(F.explode(F.arrays_zip(result.relations.result, 
                                                 result.relations.metadata)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("relation"),
                          F.expr("cols['1']['entity1']").alias("entity1"),
                          F.expr("cols['1']['entity1_begin']").alias("entity1_begin"),
                          F.expr("cols['1']['entity1_end']").alias("entity1_end"),
                          F.expr("cols['1']['chunk1']").alias("chunk1"),
                          F.expr("cols['1']['entity2']").alias("entity2"),
                          F.expr("cols['1']['entity2_begin']").alias("entity2_begin"),
                          F.expr("cols['1']['entity2_end']").alias("entity2_end"),
                          F.expr("cols['1']['chunk2']").alias("chunk2"),
                          F.expr("cols['1']['confidence']").alias("confidence")
                          )

result_df.show(50, truncate=45)

# COMMAND ----------

result_df.limit(20).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load trained model from disk

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

clinical_ner_tagger = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

loaded_re_Model = RelationExtractionModel.load("dbfs:/databricks/driver/models/custom_RE_model")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"]) \
    .setOutputCol("relations")\
    .setRelationPairs(["problem-test", "problem-treatment"])\
    .setPredictionThreshold(0.8)\
    .setMaxSyntacticDistance(3)

trained_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    clinical_ner_tagger,
    ner_chunker,
    dependency_parser,
    loaded_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

loaded_re_model = trained_pipeline.fit(empty_data)

# COMMAND ----------

text ="""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), 
one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . 
She had close follow-up with endocrinology post discharge.
"""

re_model_light = LightPipeline(loaded_re_model)

annotations = re_model_light.fullAnnotate(text)

rel_df = get_relations_df(annotations)

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC # End-to-end trained Models using BioBERT
# MAGIC #### Latest addition to Spark NLP for Heathcare - (Requires Spark NLP 2.7.3+ and Spark NLP JSL 2.7.3+)

# COMMAND ----------

# MAGIC %md
# MAGIC These models are trained as end-to-end bert models using BioBERT and ported in to the Spark NLP ecosystem. 
# MAGIC 
# MAGIC They offer SOTA performance on most benchmark tasks and outperform our existing Relation Extraction Models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ADE ReDL

# COMMAND ----------

ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")  

ade_re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(["drug-ade, ade-drug"])

ade_re_model = RelationExtractionDLModel()\
    .pretrained('redl_ade_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

ade_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_chunker,
    dependency_parser,
    ade_re_ner_chunk_filter,
    ade_re_model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

ade_model = ade_pipeline.fit(empty_data)

# COMMAND ----------

text ="""A 44-year-old man taking naproxen for chronic low back pain and a 20-year-old woman on oxaprozin for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands."""

ade_lmodel = LightPipeline(ade_model)
ade_results = ade_lmodel.fullAnnotate(text)

rel_df = get_relations_df (ade_results)

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clinical ReDL

# COMMAND ----------

clinical_ner_tagger = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

clinical_re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(4)\
    .setRelationPairs(["problem-test", "problem-treatment"])# we can set the possible relation pairs (if not set, all the relations will be calculated)
    
clinical_re_Model = RelationExtractionDLModel() \
    .pretrained('redl_clinical_biobert', "en", "clinical/models")\
    .setPredictionThreshold(0.9)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    clinical_ner_tagger,
    ner_chunker,
    dependency_parser,
    clinical_re_ner_chunk_filter,
    clinical_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)

# COMMAND ----------

text ="""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), 
one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . 
She had close follow-up with endocrinology post discharge .
"""
lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df = rel_df[(rel_df.relation!="O")]

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clinical Temporal Events ReDL

# COMMAND ----------

events_ner_tagger = MedicalNerModel.pretrained("ner_events_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")     

events_re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")
    
events_re_Model = RelationExtractionDLModel() \
    .pretrained('redl_temporal_events_biobert', "en", "clinical/models")\
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    events_ner_tagger,
    ner_chunker,
    dependency_parser,
    events_re_ner_chunk_filter,
    events_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)

# COMMAND ----------

text ="""The patient is a 68-year-old Caucasian male with past medical history of diabetes mellitus. 
He was doing fairly well until last week while mowing the lawn, he injured his right foot. """

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df.confidence = rel_df.confidence.astype(float)

rel_df = rel_df[(rel_df.relation!="O")]

rel_df[(rel_df.relation!="O")&(rel_df.entity1!=rel_df.entity2)]


# COMMAND ----------

text ="""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed.  
She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 
12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. 
"""

annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df.confidence = rel_df.confidence.astype(float)

rel_df[(rel_df.relation!="O")]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Human Phenotype - Gene ReDL

# COMMAND ----------

pgr_ner_tagger = MedicalNerModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

pgr_re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(4)
    
pgr_re_Model = RelationExtractionDLModel() \
    .pretrained('redl_human_phenotype_gene_biobert', "en", "clinical/models")\
    .setPredictionThreshold(0.6)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    pgr_ner_tagger,
    ner_chunker,
    dependency_parser,
    pgr_re_ner_chunk_filter,
    pgr_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)

# COMMAND ----------

text = "She has a hearing loss and short stature, mutations in the SH3PXD2B gene coding for the Tks4 protein are responsible for the autosomal recessive."

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df = rel_df[(rel_df.relation!=0)]

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drug-Drug Interaction ReDL

# COMMAND ----------

ddi_ner_tagger = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   

ddi_re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(4)
    
ddi_re_Model = RelationExtractionDLModel() \
    .pretrained('redl_drug_drug_interaction_biobert', "en", "clinical/models")\
    .setPredictionThreshold(0.9)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ddi_ner_tagger,
    ner_chunker,
    dependency_parser,
    ddi_re_ner_chunk_filter,
    ddi_re_Model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)

# COMMAND ----------

text='When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced. \
If additional adrenergic drugs are to be administered by any route, \
they should be used with caution because the pharmacologically predictable sympathetic effects of Metformin may be potentiated'

lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drug-Protein Interaction ReDL

# COMMAND ----------

drugprot_ner_tagger = MedicalNerModel.pretrained("ner_drugprot_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   

ner_converter = NerConverter()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

# Set a filter on pairs of named entities which will be treated as relation candidates
drugprot_re_ner_chunk_filter = RENerChunksFilter()\
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(4)\
   .setRelationPairs(["checmical-gene", "chemical-gene_and_chemical", "gene_and_chemical-gene"])
    
drugprot_re_Model = RelationExtractionDLModel()\
    .pretrained('redl_drugprot_biobert', "en", "clinical/models")\
    .setPredictionThreshold(0.9)\
    .setInputCols(["re_ner_chunks", "sentences"])\
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, drugprot_ner_tagger, ner_converter, pos_tagger, dependency_parser, drugprot_re_ner_chunk_filter, drugprot_re_Model])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

# COMMAND ----------

text='''Lipid specific activation of the murine P4-ATPase Atp8a1 (ATPase II). The asymmetric transbilayer distribution of phosphatidylserine (PS) in the mammalian plasma membrane and secretory vesicles is maintained, in part, by an ATP-dependent transporter. This aminophospholipid "flippase" selectively transports PS to the cytosolic leaflet of the bilayer and is sensitive to vanadate, Ca(2+), and modification by sulfhydryl reagents. Although the flippase has not been positively identified, a subfamily of P-type ATPases has been proposed to function as transporters of amphipaths, including PS and other phospholipids. A candidate PS flippase ATP8A1 (ATPase II), originally isolated from bovine secretory vesicles, is a member of this subfamily based on sequence homology to the founding member of the subfamily, the yeast protein Drs2, which has been linked to ribosomal assembly, the formation of Golgi-coated vesicles, and the maintenance of PS asymmetry. To determine if ATP8A1 has biochemical characteristics consistent with a PS flippase, a murine homologue of this enzyme was expressed in insect cells and purified. The purified Atp8a1 is inactive in detergent micelles or in micelles containing phosphatidylcholine, phosphatidic acid, or phosphatidylinositol, is minimally activated by phosphatidylglycerol or phosphatidylethanolamine (PE), and is maximally activated by PS. The selectivity for PS is dependent upon multiple elements of the lipid structure. Similar to the plasma membrane PS transporter, Atp8a1 is activated only by the naturally occurring sn-1,2-glycerol isomer of PS and not the sn-2,3-glycerol stereoisomer. Both flippase and Atp8a1 activities are insensitive to the stereochemistry of the serine headgroup. Most modifications of the PS headgroup structure decrease recognition by the plasma membrane PS flippase. Activation of Atp8a1 is also reduced by these modifications; phosphatidylserine-O-methyl ester, lysophosphatidylserine, glycerophosphoserine, and phosphoserine, which are not transported by the plasma membrane flippase, do not activate Atp8a1. Weakly translocated lipids (PE, phosphatidylhydroxypropionate, and phosphatidylhomoserine) are also weak Atp8a1 activators. However, N-methyl-phosphatidylserine, which is transported by the plasma membrane flippase at a rate equivalent to PS, is incapable of activating Atp8a1 activity. These results indicate that the ATPase activity of the secretory granule Atp8a1 is activated by phospholipids binding to a specific site whose properties (PS selectivity, dependence upon glycerol but not serine, stereochemistry, and vanadate sensitivity) are similar to, but distinct from, the properties of the substrate binding site of the plasma membrane flippase.'''

lmodel = LightPipeline(model)

results = lmodel.fullAnnotate(text)

# COMMAND ----------

results[0].keys()

# COMMAND ----------

results[0]["ner_chunks"]

# COMMAND ----------

results[0]["relations"]

# COMMAND ----------

rel_df = get_relations_df (results)

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chemical–Protein Interactions (ChemProt) ReDL

# COMMAND ----------

chemprot_ner_tagger = MedicalNerModel.pretrained("ner_chemprot_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   

chemprot_re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(4)
    
chemprot_re_Model = RelationExtractionDLModel() \
    .pretrained('redl_chemprot_biobert', "en", "clinical/models")\
    .setPredictionThreshold(0.9)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    chemprot_ner_tagger,
    ner_chunker,
    dependency_parser,
    chemprot_re_ner_chunk_filter,
    chemprot_re_Model
])


empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)

# COMMAND ----------

text='''
In this study, we examined the effects of mitiglinide on various cloned K(ATP) channels (Kir6.2/SUR1, Kir6.2/SUR2A, and Kir6.2/SUR2B) reconstituted in COS-1 cells, and compared them to another meglitinide-related compound, nateglinide. Patch-clamp analysis using inside-out recording configuration showed that mitiglinide inhibits the Kir6.2/SUR1 channel currents in a dose-dependent manner (IC50 value, 100 nM) but does not significantly inhibit either Kir6.2/SUR2A or Kir6.2/SUR2B channel currents even at high doses (more than 10 microM). Nateglinide inhibits Kir6.2/SUR1 and Kir6.2/SUR2B channels at 100 nM, and inhibits Kir6.2/SUR2A channels at high concentrations (1 microM). Binding experiments on mitiglinide, nateglinide, and repaglinide to SUR1 expressed in COS-1 cells revealed that they inhibit the binding of [3H]glibenclamide to SUR1 (IC50 values: mitiglinide, 280 nM; nateglinide, 8 microM; repaglinide, 1.6 microM), suggesting that they all share a glibenclamide binding site. The insulin responses to glucose, mitiglinide, tolbutamide, and glibenclamide in MIN6 cells after chronic mitiglinide, nateglinide, or repaglinide treatment were comparable to those after chronic tolbutamide and glibenclamide treatment. These results indicate that, similar to the sulfonylureas, mitiglinide is highly specific to the Kir6.2/SUR1 complex, i.e., the pancreatic beta-cell K(ATP) channel, and suggest that mitiglinide may be a clinically useful anti-diabetic drug.
'''
lmodel = LightPipeline(model)
annotations = lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)

rel_df[rel_df.entity1!=rel_df.entity2]

# COMMAND ----------

# MAGIC %md
# MAGIC # Relation Extraction Across Sentences
# MAGIC 
# MAGIC We can extract relations across the sentences in the document by dropping the `SentenceDetector` in the pipeline. Since ReDL models are trained with BERT embeddings, they can perform better than their RE versions. **BUT** it all depends on the data that we are working on, so we can try different models and use the best one in our pipeline.

# COMMAND ----------

import pandas as pd

def get_only_relations_df(results, col='relations'):
  rel_pairs=[]
  for rel in results[0][col]:
      rel_pairs.append((
          rel.metadata['entity1_begin'],
          rel.metadata['entity1_end'],
          rel.metadata['chunk1'], 
          rel.metadata['entity1'], 
          rel.metadata['entity2_begin'],
          rel.metadata['entity2_end'],
          rel.metadata['chunk2'], 
          rel.metadata['entity2'],
          rel.result, 
          rel.metadata['confidence']
      ))

  rel_df = pd.DataFrame(rel_pairs, columns=['entity1_begin','entity1_end','chunk1', 'entity1', 'entity2_begin','entity2_end','chunk2', 'entity2', 'relation', 'confidence'])

  return rel_df

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["document", "tokens"])\
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models")\
    .setInputCols("document", "tokens", "embeddings")\
    .setOutputCol("ner_tags")

ner_chunker = NerConverterInternal()\
    .setInputCols(["document", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["document", "tokens"])\
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["document", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

ade_re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(0)\
    .setRelationPairs(["drug-ade, ade-drug"])

ade_re_model = RelationExtractionDLModel()\
    .pretrained('redl_ade_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "document"]) \
    .setOutputCol("relations")

ade_pipeline = Pipeline(stages=[
    documenter,
    tokenizer, 
    words_embedder, 
    ner_tagger,
    ner_chunker,
    pos_tagger, 
    dependency_parser,
    ade_re_ner_chunk_filter,
    ade_re_model
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
ade_model = ade_pipeline.fit(empty_data)

# COMMAND ----------

ade_lmodel = LightPipeline(ade_model)

text ="""A 44-year-old man taking naproxen for chronic low back pain and a 20-year-old woman on oxaprozin for rheumatoid arthritis. 
They both presented with tense bullae and cutaneous fragility on the face and the back of the hands. 
"""

ade_results = ade_lmodel.fullAnnotate(text)
rel_df = get_only_relations_df (ade_results)

rel_df

# COMMAND ----------

text = """The patient continued to receive regular insulin 4 times per day over the following 3 years with only occasional hives. 
Two days ago, suddenly severe urticaria, angioedema, and occasional wheezing began."""

ade_results = ade_lmodel.fullAnnotate(text)
rel_df = get_only_relations_df (ade_results)

rel_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Merging Multiple RE Model Results

# COMMAND ----------

# MAGIC %md
# MAGIC We can merge multiple RE model results by using `AnnotationMerger()`. <br/>
# MAGIC Now, we will build a pipeline consisting of `posology_re`, `re_ade_clinical` models and `AnnotationMerger()` to merge these RE models' results.

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

pos_ner_tagger = MedicalNerModel()\
    .pretrained("ner_posology", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_pos")

pos_ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ner_pos"])\
    .setOutputCol("pos_ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

pos_reModel = RelationExtractionModel()\
    .pretrained("posology_re")\
    .setInputCols(["embeddings", "pos_tags", "pos_ner_chunks", "dependencies"])\
    .setOutputCol("pos_relations")\
    .setMaxSyntacticDistance(4)

ade_ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ade_ner_tags")  

ade_ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ade_ner_tags"])\
    .setOutputCol("ade_ner_chunks")

ade_reModel = RelationExtractionModel()\
    .pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ade_ner_chunks", "dependencies"])\
    .setOutputCol("ade_relations")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(["drug-ade, ade-drug"])

annotation_merger = AnnotationMerger()\
    .setInputCols("ade_relations", "pos_relations")\
    .setInputType("category")\
    .setOutputCol("all_relations")

merger_pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    pos_ner_tagger,
    pos_ner_chunker,
    dependency_parser,
    pos_reModel,
    ade_ner_tagger,
    ade_ner_chunker,
    ade_reModel,
    annotation_merger
])

empty_df= spark.createDataFrame([[""]]).toDF("text")
merger_model= merger_pipeline.fit(empty_df)

# COMMAND ----------

text = """
The patient was prescribed 1 unit of naproxen for 5 days after meals for chronic low back pain. The patient was also given 1 unit of oxaprozin daily for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands.. 
"""
data = spark.createDataFrame([[text]]).toDF("text")

result = merger_model.transform(data)
result.show()

# COMMAND ----------

from pyspark.sql import functions as F

result_df = result.select(F.explode(F.arrays_zip(result.pos_relations.result,
                                                 result.pos_relations.metadata,
                                                 result.ade_relations.result,
                                                 result.ade_relations.metadata,
                                                 result.all_relations.result,
                                                 result.all_relations.metadata)).alias("cols")) \
                  .select(
                          F.expr("cols['0']").alias("pos_relations"),\
                          F.expr("cols['1']['entity1']").alias("pos_relations_entity1"),\
                          F.expr("cols['1']['chunk1']" ).alias("pos_relations_chunk1" ),\
                          F.expr("cols['1']['entity2']").alias("pos_relations_entity2"),\
                          F.expr("cols['1']['chunk2']" ).alias("pos_relations_chunk2" ),\
                          F.expr("cols['2']").alias("ade_relations"),\
                          F.expr("cols['3']['entity1']").alias("ade_relations_entity1"),\
                          F.expr("cols['3']['chunk1']" ).alias("ade_relations_chunk1" ),\
                          F.expr("cols['3']['entity2']").alias("ade_relations_entity2"),\
                          F.expr("cols['3']['chunk2']" ).alias("ade_relations_chunk2" ),\
                          F.expr("cols['4']").alias("all_relations"),\
                          F.expr("cols['5']['entity1']").alias("all_relations_entity1"),\
                          F.expr("cols['5']['chunk1']" ).alias("all_relations_chunk1" ),\
                          F.expr("cols['5']['entity2']").alias("all_relations_entity2"),\
                          F.expr("cols['5']['chunk2']" ).alias("all_relations_chunk2" )
                          )

result_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we will build a LightPipeline and visualize the results by using `RelationExtractionVisualizer`

# COMMAND ----------

lp_merger = LightPipeline(merger_model)
lp_res = lp_merger.fullAnnotate(text)[0]

# COMMAND ----------

from sparknlp_display import RelationExtractionVisualizer


visualizer = RelationExtractionVisualizer()
vis = visualizer.display(lp_res, 'all_relations', show_relations=True, return_html=True) # default show_relations: True

displayHTML(vis)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clinical Knowledge Graph Creation using Neo4j

# COMMAND ----------

# MAGIC %md
# MAGIC We have recently published a [Medium article](https://medium.com/spark-nlp/creating-knowledge-graph-by-spark-nlp-neo4j-9d18706aa08b) explains "How to create a  clinical knowledge graph using Spark NLP and Neo4j." and for detailed information, you can play with [the notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.2.Clinical_RE_Knowledge_Graph_with_Neo4j.ipynb). Briefly, in this article, we built a Knowledge Graph (KG) using Spark NLP Relation Extraction (RE) Models and Neo4j. Some of the Spark NLP users ask "How can I use the RE results?". First, you can use the results to feed downstream pipelines. Secondly, you can create a KG to try to get insights. The article explains how to accomplish the second option.

# COMMAND ----------

# MAGIC %md
# MAGIC # Relation Extraction Model-NER Model-Relation Pairs Table
# MAGIC 
# MAGIC In the table below, you can find the Relationship Extraction models and the most appropriate NER models and Relationship Pairs that can be used to get the most efficient results when using these models. <br/>
# MAGIC Also, you can reach this table in [Sparknlp for Healthcare Documentation](https://nlp.johnsnowlabs.com/docs/en/best_practices_pretrained_models#relation-extraction-models-and-relation-pairs-table)

# COMMAND ----------

# MAGIC %md
# MAGIC |    | RELATION EXTRACTION MODEL            | RELATION EXTRACTION MODEL LABELS                                        | NER MODEL                         | RELATION PAIRS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# MAGIC |---:|:-------------------------------------|:------------------------------------------------------------------------|:----------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC |  0 | re_bodypart_proceduretest            | 0-1                                                                     | ner_jsl                           | ["external_body_part_or_region-test", <br/> "test-external_body_part_or_region",<br/>"internal_organ_or_component-test",<br/>"test-internal_organ_or_component",<br/>"external_body_part_or_region-procedure",<br/>"procedure-external_body_part_or_region",<br/>"procedure-internal_organ_or_component",<br/>"internal_organ_or_component-procedure"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC |  1 | re_ade_clinical                      | 0-1                                                                     | ner_ade_clinical                  | ["ade-drug",<br/> "drug-ade"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# MAGIC |  2 | redl_chemprot_biobert                | CPR:1, CPR:2, CPR:3, CPR:4, CPR:5, CPR:6, CPR:7, CPR:8, CPR:9, CPR:10   | ner_chemprot_clinical             |   No need to set pairs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC |  3 | re_human_phenotype_gene_clinical     | 0-1                                                                     | ner_human_phenotype_gene_clinical |   No need to set pairs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC |  4 | re_bodypart_directions               | 0-1                                                                     | ner_jsl                           | ["direction-external_body_part_or_region",<br/>"external_body_part_or_region-direction",<br/>"direction-internal_organ_or_component",<br/>"internal_organ_or_component-direction"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# MAGIC |  5 | re_bodypart_problem                  | 0-1                                                                     | ner_jsl                           | ["internal_organ_or_component-cerebrovascular_disease", "cerebrovascular_disease-internal_organ_or_component",<br/>"internal_organ_or_component-communicable_disease", "communicable_disease-internal_organ_or_component",<br/>"internal_organ_or_component-diabetes", "diabetes-internal_organ_or_component",<br/>"internal_organ_or_component-disease_syndrome_disorder", "disease_syndrome_disorder-internal_organ_or_component",<br/>"internal_organ_or_component-ekg_findings", "ekg_findings-internal_organ_or_component",<br/>"internal_organ_or_component-heart_disease", "heart_disease-internal_organ_or_component",<br/>"internal_organ_or_component-hyperlipidemia", "hyperlipidemia-internal_organ_or_component",<br/>"internal_organ_or_component-hypertension", "hypertension-internal_organ_or_component",<br/>"internal_organ_or_component-imagingfindings", "imagingfindings-internal_organ_or_component",<br/>"internal_organ_or_component-injury_or_poisoning", "injury_or_poisoning-internal_organ_or_component",<br/>"internal_organ_or_component-kidney_disease", "kidney_disease-internal_organ_or_component",<br/>"internal_organ_or_component-oncological", "oncological-internal_organ_or_component",<br/>"internal_organ_or_component-psychological_condition", "psychological_condition-internal_organ_or_component",<br/>"internal_organ_or_component-symptom", "symptom-internal_organ_or_component",<br/>"internal_organ_or_component-vs_finding", "vs_finding-internal_organ_or_component",<br/>"external_body_part_or_region-communicable_disease", "communicable_disease-external_body_part_or_region",<br/>"external_body_part_or_region-diabetes", "diabetes-external_body_part_or_region",<br/>"external_body_part_or_region-disease_syndrome_disorder", "disease_syndrome_disorder-external_body_part_or_region",<br/>"external_body_part_or_region-hypertension", "hypertension-external_body_part_or_region",<br/>"external_body_part_or_region-imagingfindings", "imagingfindings-external_body_part_or_region",<br/>"external_body_part_or_region-injury_or_poisoning", "injury_or_poisoning-external_body_part_or_region",<br/>"external_body_part_or_region-obesity", "obesity-external_body_part_or_region",<br/>"external_body_part_or_region-oncological", "oncological-external_body_part_or_region",<br/>"external_body_part_or_region-overweight", "overweight-external_body_part_or_region",<br/>"external_body_part_or_region-symptom", "symptom-external_body_part_or_region",<br/>"external_body_part_or_region-vs_finding", "vs_finding-external_body_part_or_region"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# MAGIC |  6 | re_drug_drug_interaction_clinical    | DDI-advise, DDI-effect, DDI-mechanism, DDI-int, DDI-false               | ner_posology                      | ["drug-drug"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# MAGIC |  7 | re_clinical                          | TrIP, TrWP, TrCP, TrAP, TrAP, TeRP, TeCP, PIP                           | ner_clinical                      |   No need to set pairs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC |  8 | re_temporal_events_clinical          | AFTER, BEFORE, OVERLAP                                                  | ner_events_clinical               |   No need to set pairs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC |  9 | re_temporal_events_enriched_clinical | BEFORE, AFTER, SIMULTANEOUS, BEGUN_BY, ENDED_BY, DURING, BEFORE_OVERLAP | ner_events_clinical               |   No need to set pairs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC | 10 | re_test_problem_finding              | 0-1                                                                     | ner_jsl                           | ["test-cerebrovascular_disease", "cerebrovascular_disease-test",<br/>"test-communicable_disease", "communicable_disease-test",<br/>"test-diabetes", "diabetes-test",<br/>"test-disease_syndrome_disorder", "disease_syndrome_disorder-test",<br/>"test-heart_disease", "heart_disease-test",<br/>"test-hyperlipidemia", "hyperlipidemia-test",<br/>"test-hypertension", "hypertension-test",<br/>"test-injury_or_poisoning", "injury_or_poisoning-test",<br/>"test-kidney_disease", "kidney_disease-test",<br/>"test-obesity", "obesity-test",<br/>"test-oncological", "oncological-test",<br/>"test-psychological_condition", "psychological_condition-test",<br/>"test-symptom", "symptom-test",<br/>"ekg_findings-disease_syndrome_disorder", "disease_syndrome_disorder-ekg_findings",<br/>"ekg_findings-heart_disease", "heart_disease-ekg_findings",<br/>"ekg_findings-symptom", "symptom-ekg_findings",<br/>"imagingfindings-cerebrovascular_disease", "cerebrovascular_disease-imagingfindings",<br/>"imagingfindings-communicable_disease", "communicable_disease-imagingfindings",<br/>"imagingfindings-disease_syndrome_disorder", "disease_syndrome_disorder-imagingfindings",<br/>"imagingfindings-heart_disease", "heart_disease-imagingfindings",<br/>"imagingfindings-hyperlipidemia", "hyperlipidemia-imagingfindings",<br/>"imagingfindings-hypertension", "hypertension-imagingfindings",<br/>"imagingfindings-injury_or_poisoning", "injury_or_poisoning-imagingfindings",<br/>"imagingfindings-kidney_disease", "kidney_disease-imagingfindings",<br/>"imagingfindings-oncological", "oncological-imagingfindings",<br/>"imagingfindings-psychological_condition", "psychological_condition-imagingfindings",<br/>"imagingfindings-symptom", "symptom-imagingfindings",<br/>"vs_finding-cerebrovascular_disease", "cerebrovascular_disease-vs_finding",<br/>"vs_finding-communicable_disease", "communicable_disease-vs_finding",<br/>"vs_finding-diabetes", "diabetes-vs_finding",<br/>"vs_finding-disease_syndrome_disorder", "disease_syndrome_disorder-vs_finding",<br/>"vs_finding-heart_disease", "heart_disease-vs_finding",<br/>"vs_finding-hyperlipidemia", "hyperlipidemia-vs_finding",<br/>"vs_finding-hypertension", "hypertension-vs_finding",<br/>"vs_finding-injury_or_poisoning", "injury_or_poisoning-vs_finding",<br/>"vs_finding-kidney_disease", "kidney_disease-vs_finding",<br/>"vs_finding-obesity", "obesity-vs_finding",<br/>"vs_finding-oncological", "oncological-vs_finding",<br/>"vs_finding-overweight", "overweight-vs_finding",<br/>"vs_finding-psychological_condition", "psychological_condition-vs_finding",<br/>"vs_finding-symptom", "symptom-vs_finding"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# MAGIC | 11 | re_test_result_date                  | is_finding_of, is_result_of, is_date_of, O                              | ner_jsl                           | ["test-test_result", "test_result-test",<br/>"test-date", "date-test",<br/>"test-imagingfindings", "imagingfindings-test",<br/>"test-ekg_findings", "ekg_findings-test",<br/>"date-test_result", "test_result-date",<br/>"date-imagingfindings", "imagingfindings-date",<br/>"date-ekg_findings", "ekg_findings-date"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# MAGIC | 12 | re_date_clinical                     | 0-1                                                                     | ner_jsl                           | ["date-admission_discharge", "admission_discharge-date",<br/>"date-alcohol", "alcohol-date",<br/>"date-allergen", "allergen-date",<br/>"date-bmi", "bmi-date",<br/>"date-birth_entity", "birth_entity-date",<br/>"date-blood_pressure", "blood_pressure-date",<br/>"date-cerebrovascular_disease", "cerebrovascular_disease-date",<br/>"date-clinical_dept", "clinical_dept-date",<br/>"date-communicable_disease", "communicable_disease-date",<br/>"date-death_entity", "death_entity-date",<br/>"date-diabetes", "diabetes-date",<br/>"date-diet", "diet-date",<br/>"date-disease_syndrome_disorder", "disease_syndrome_disorder-date",<br/>"date-drug_brandname", "drug_brandname-date",<br/>"date-drug_ingredient", "drug_ingredient-date",<br/>"date-ekg_findings", "ekg_findings-date",<br/>"date-external_body_part_or_region", "external_body_part_or_region-date",<br/>"date-fetus_newborn", "fetus_newborn-date",<br/>"date-hdl", "hdl-date",<br/>"date-heart_disease", "heart_disease-date",<br/>"date-height", "height-date",<br/>"date-hyperlipidemia", "hyperlipidemia-date",<br/>"date-hypertension", "hypertension-date",<br/>"date-imagingfindings", "imagingfindings-date",<br/>"date-imaging_technique", "imaging_technique-date",<br/>"date-injury_or_poisoning", "injury_or_poisoning-date",<br/>"date-internal_organ_or_component", "internal_organ_or_component-date",<br/>"date-kidney_disease", "kidney_disease-date",<br/>"date-ldl", "ldl-date",<br/>"date-modifier", "modifier-date",<br/>"date-o2_saturation", "o2_saturation-date",<br/>"date-obesity", "obesity-date",<br/>"date-oncological", "oncological-date",<br/>"date-overweight", "overweight-date",<br/>"date-oxygen_therapy", "oxygen_therapy-date",<br/>"date-pregnancy", "pregnancy-date",<br/>"date-procedure", "procedure-date",<br/>"date-psychological_condition", "psychological_condition-date",<br/>"date-pulse", "pulse-date",<br/>"date-respiration", "respiration-date",<br/>"date-smoking", "smoking-date",<br/>"date-substance", "substance-date",<br/>"date-substance_quantity", "substance_quantity-date",<br/>"date-symptom", "symptom-date",<br/>"date-temperature", "temperature-date",<br/>"date-test", "test-date",<br/>"date-test_result", "test_result-date",<br/>"date-total_cholesterol", "total_cholesterol-date",<br/>"date-treatment", "treatment-date",<br/>"date-triglycerides", "triglycerides-date",<br/>"date-vs_finding", "vs_finding-date",<br/>"date-vaccine", "vaccine-date",<br/>"date-vital_signs_header", "vital_signs_header-date",<br/>"date-weight", "weight-date",<br/>"time-admission_discharge", "admission_discharge-time",<br/>"time-alcohol", "alcohol-time",<br/>"time-allergen", "allergen-time",<br/>"time-bmi", "bmi-time",<br/>"time-birth_entity", "birth_entity-time",<br/>"time-blood_pressure", "blood_pressure-time",<br/>"time-cerebrovascular_disease", "cerebrovascular_disease-time",<br/>"time-clinical_dept", "clinical_dept-time",<br/>"time-communicable_disease", "communicable_disease-time",<br/>"time-death_entity", "death_entity-time",<br/>"time-diabetes", "diabetes-time",<br/>"time-diet", "diet-time",<br/>"time-disease_syndrome_disorder", "disease_syndrome_disorder-time",<br/>"time-drug_brandname", "drug_brandname-time",<br/>"time-drug_ingredient", "drug_ingredient-time",<br/>"time-ekg_findings", "ekg_findings-time",<br/>"time-external_body_part_or_region", "external_body_part_or_region-time",<br/>"time-fetus_newborn", "fetus_newborn-time",<br/>"time-hdl", "hdl-time",<br/>"time-heart_disease", "heart_disease-time",<br/>"time-height", "height-time",<br/>"time-hyperlipidemia", "hyperlipidemia-time",<br/>"time-hypertension", "hypertension-time",<br/>"time-imagingfindings", "imagingfindings-time",<br/>"time-imaging_technique", "imaging_technique-time",<br/>"time-injury_or_poisoning", "injury_or_poisoning-time",<br/>"time-internal_organ_or_component", "internal_organ_or_component-time",<br/>"time-kidney_disease", "kidney_disease-time",<br/>"time-ldl", "ldl-time",<br/>"time-modifier", "modifier-time",<br/>"time-o2_saturation", "o2_saturation-time",<br/>"time-obesity", "obesity-time",<br/>"time-oncological", "oncological-time",<br/>"time-overweight", "overweight-time",<br/>"time-oxygen_therapy", "oxygen_therapy-time",<br/>"time-pregnancy", "pregnancy-time",<br/>"time-procedure", "procedure-time",<br/>"time-psychological_condition", "psychological_condition-time",<br/>"time-pulse", "pulse-time",<br/>"time-respiration", "respiration-time",<br/>"time-smoking", "smoking-time",<br/>"time-substance", "substance-time",<br/>"time-substance_quantity", "substance_quantity-time",<br/>"time-symptom", "symptom-time",<br/>"time-temperature", "temperature-time",<br/>"time-test", "test-time",<br/>"time-test_result", "test_result-time",<br/>"time-total_cholesterol", "total_cholesterol-time",<br/>"time-treatment", "treatment-time",<br/>"time-triglycerides", "triglycerides-time",<br/>"time-vs_finding", "vs_finding-time",<br/>"time-vaccine", "vaccine-time",<br/>"time-vital_signs_header", "vital_signs_header-time",<br/>"time-weight", "weight-time",<br/>"relativedate-admission_discharge", "admission_discharge-relativedate",<br/>"relativedate-alcohol", "alcohol-relativedate",<br/>"relativedate-allergen", "allergen-relativedate",<br/>"relativedate-bmi", "bmi-relativedate",<br/>"relativedate-birth_entity", "birth_entity-relativedate",<br/>"relativedate-blood_pressure", "blood_pressure-relativedate",<br/>"relativedate-cerebrovascular_disease", "cerebrovascular_disease-relativedate",<br/>"relativedate-clinical_dept", "clinical_dept-relativedate",<br/>"relativedate-communicable_disease", "communicable_disease-relativedate",<br/>"relativedate-death_entity", "death_entity-relativedate",<br/>"relativedate-diabetes", "diabetes-relativedate",<br/>"relativedate-diet", "diet-relativedate",<br/>"relativedate-disease_syndrome_disorder", "disease_syndrome_disorder-relativedate",<br/>"relativedate-drug_brandname", "drug_brandname-relativedate",<br/>"relativedate-drug_ingredient", "drug_ingredient-relativedate",<br/>"relativedate-ekg_findings", "ekg_findings-relativedate",<br/>"relativedate-external_body_part_or_region", "external_body_part_or_region-relativedate",<br/>"relativedate-fetus_newborn", "fetus_newborn-relativedate",<br/>"relativedate-hdl", "hdl-relativedate",<br/>"relativedate-heart_disease", "heart_disease-relativedate",<br/>"relativedate-height", "height-relativedate",<br/>"relativedate-hyperlipidemia", "hyperlipidemia-relativedate",<br/>"relativedate-hypertension", "hypertension-relativedate",<br/>"relativedate-imagingfindings", "imagingfindings-relativedate",<br/>"relativedate-imaging_technique", "imaging_technique-relativedate",<br/>"relativedate-injury_or_poisoning", "injury_or_poisoning-relativedate",<br/>"relativedate-internal_organ_or_component", "internal_organ_or_component-relativedate",<br/>"relativedate-kidney_disease", "kidney_disease-relativedate",<br/>"relativedate-ldl", "ldl-relativedate",<br/>"relativedate-modifier", "modifier-relativedate",<br/>"relativedate-o2_saturation", "o2_saturation-relativedate",<br/>"relativedate-obesity", "obesity-relativedate",<br/>"relativedate-oncological", "oncological-relativedate",<br/>"relativedate-overweight", "overweight-relativedate",<br/>"relativedate-oxygen_therapy", "oxygen_therapy-relativedate",<br/>"relativedate-pregnancy", "pregnancy-relativedate",<br/>"relativedate-procedure", "procedure-relativedate",<br/>"relativedate-psychological_condition", "psychological_condition-relativedate",<br/>"relativedate-pulse", "pulse-relativedate",<br/>"relativedate-respiration", "respiration-relativedate",<br/>"relativedate-smoking", "smoking-relativedate",<br/>"relativedate-substance", "substance-relativedate",<br/>"relativedate-substance_quantity", "substance_quantity-relativedate",<br/>"relativedate-symptom", "symptom-relativedate",<br/>"relativedate-temperature", "temperature-relativedate",<br/>"relativedate-test", "test-relativedate",<br/>"relativedate-test_result", "test_result-relativedate",<br/>"relativedate-total_cholesterol", "total_cholesterol-relativedate",<br/>"relativedate-treatment", "treatment-relativedate",<br/>"relativedate-triglycerides", "triglycerides-relativedate",<br/>"relativedate-vs_finding", "vs_finding-relativedate",<br/>"relativedate-vaccine", "vaccine-relativedate",<br/>"relativedate-vital_signs_header", "vital_signs_header-relativedate",<br/>"relativedate-weight", "weight-relativedate",<br/>"relativetime-admission_discharge", "admission_discharge-relativetime",<br/>"relativetime-alcohol", "alcohol-relativetime",<br/>"relativetime-allergen", "allergen-relativetime",<br/>"relativetime-bmi", "bmi-relativetime",<br/>"relativetime-birth_entity", "birth_entity-relativetime",<br/>"relativetime-blood_pressure", "blood_pressure-relativetime",<br/>"relativetime-cerebrovascular_disease", "cerebrovascular_disease-relativetime",<br/>"relativetime-clinical_dept", "clinical_dept-relativetime",<br/>"relativetime-communicable_disease", "communicable_disease-relativetime",<br/>"relativetime-death_entity", "death_entity-relativetime",<br/>"relativetime-diabetes", "diabetes-relativetime",<br/>"relativetime-diet", "diet-relativetime",<br/>"relativetime-disease_syndrome_disorder", "disease_syndrome_disorder-relativetime",<br/>"relativetime-drug_brandname", "drug_brandname-relativetime",<br/>"relativetime-drug_ingredient", "drug_ingredient-relativetime",<br/>"relativetime-ekg_findings", "ekg_findings-relativetime",<br/>"relativetime-external_body_part_or_region", "external_body_part_or_region-relativetime",<br/>"relativetime-fetus_newborn", "fetus_newborn-relativetime",<br/>"relativetime-hdl", "hdl-relativetime",<br/>"relativetime-heart_disease", "heart_disease-relativetime",<br/>"relativetime-height", "height-relativetime",<br/>"relativetime-hyperlipidemia", "hyperlipidemia-relativetime",<br/>"relativetime-hypertension", "hypertension-relativetime",<br/>"relativetime-imagingfindings", "imagingfindings-relativetime",<br/>"relativetime-imaging_technique", "imaging_technique-relativetime",<br/>"relativetime-injury_or_poisoning", "injury_or_poisoning-relativetime",<br/>"relativetime-internal_organ_or_component", "internal_organ_or_component-relativetime",<br/>"relativetime-kidney_disease", "kidney_disease-relativetime",<br/>"relativetime-ldl", "ldl-relativetime",<br/>"relativetime-modifier", "modifier-relativetime",<br/>"relativetime-o2_saturation", "o2_saturation-relativetime",<br/>"relativetime-obesity", "obesity-relativetime",<br/>"relativetime-oncological", "oncological-relativetime",<br/>"relativetime-overweight", "overweight-relativetime",<br/>"relativetime-oxygen_therapy", "oxygen_therapy-relativetime",<br/>"relativetime-pregnancy", "pregnancy-relativetime",<br/>"relativetime-procedure", "procedure-relativetime",<br/>"relativetime-psychological_condition", "psychological_condition-relativetime",<br/>"relativetime-pulse", "pulse-relativetime",<br/>"relativetime-respiration", "respiration-relativetime",<br/>"relativetime-smoking", "smoking-relativetime",<br/>"relativetime-substance", "substance-relativetime",<br/>"relativetime-substance_quantity", "substance_quantity-relativetime",<br/>"relativetime-symptom", "symptom-relativetime",<br/>"relativetime-temperature", "temperature-relativetime",<br/>"relativetime-test", "test-relativetime",<br/>"relativetime-test_result", "test_result-relativetime",<br/>"relativetime-total_cholesterol", "total_cholesterol-relativetime",<br/>"relativetime-treatment", "treatment-relativetime",<br/>"relativetime-triglycerides", "triglycerides-relativetime",<br/>"relativetime-vs_finding", "vs_finding-relativetime",<br/>"relativetime-vaccine", "vaccine-relativetime",<br/>"relativetime-vital_signs_header", "vital_signs_header-relativetime",<br/>"relativetime-weight", "weight-relativetime"] |

# COMMAND ----------

