# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Advanced Sentence Detection using Pretrained Classification Model

# COMMAND ----------

# MAGIC %md
# MAGIC `SentenceDetectorDL` (SDDL) is based on a general-purpose neural network model for sentence boundary detection.  The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.
# MAGIC 
# MAGIC In this model, we treated the sentence boundary detection task as a classification problem using a DL CNN architecture. We also modified the original implemenation a little bit to cover broken sentences and some impossible end of line chars.

# COMMAND ----------

import os
import json
import sys
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

pd.set_option('display.max_columns', 100)  
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 800)
pd.set_option('display.expand_frame_repr', False)

print("Spark NLP Version :", sparknlp.version())
print('Spark NLP_JSL Version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencerDL_hc = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
	.setInputCols(["document"]) \
	.setOutputCol("sentences")

sd_pipeline_hc = PipelineModel(stages=[documenter, sentencerDL_hc])
sd_model_hc = LightPipeline(sd_pipeline_hc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## *SetenceDetectorDL_HC* Performance and Comparison with *Spacy Sentence Splitter* on different *Clinical Texts*

# COMMAND ----------

# Select one of the medical texts given above for the prediction
def get_sentences_sddl(text):
    data = []
    for anno in sd_model_hc.fullAnnotate(text)[0]["sentences"]:
        data.append((anno.metadata["sentence"], anno.begin, anno.end, anno.result))   
    sentences_df = pd.DataFrame(data, columns=["sentence","begin", "end", "result" ])
    return sentences_df.style.set_properties(**{'text-align': 'left', 'border-color':'Black','border-width':'thin','border-style':'solid'})

# COMMAND ----------

!pip install spacy
!python3 -m spacy download en_core_web_sm

# COMMAND ----------

import sys

sys.path.append('/databricks/driver/')
sys.path.append('/databricks/driver/spacy.py')
sys.path.append('/databricks/driver/en_core_web_sm.py')

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# COMMAND ----------

def get_sentences_spacy(text):
   
    data = []
    for i,sent in enumerate(nlp(text).sents):
        data.append((i, sent))

    sentences_df = pd.DataFrame(data, columns=["sentence","result" ])
    
    return sentences_df.style.set_properties(**{'text-align': 'left', 'border-color':'Black','border-width':'thin','border-style':'solid'})

# COMMAND ----------

text_1 = '''He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.Repleted with 20 meq kcl po, 30 mmol K-phos iv and 2 gms mag so4 iv. LASIX CHANGED TO 40 PO BID WHICH IS SAME AS HE TAKES AT HOME - RECEIVED 40 PO IN AM - 700CC U/O TOTAL FOR FLUID NEGATIVE ~ 600 THUS FAR TODAY, ~ 600 NEG LOS. pt initially hypertensive 160s-180s gave prn doses if IV hydralazine without effect, labetalol increased from 100 mg to 200 mg PO TID which kept SBP > 160 for rest of night.Transferred to the EW found to be hypertensive BP 253/167 HR 132 ST treated with IV NTG gtt 20-120mcg/ and captopril 12.5mg sl x3. During the day pt's resp status has been very tenuous, responded to lasix in the am but then became hypotensive around 1800 tx with 500cc NS bolus and a unit of RBC did improve her BP to 90-100/60 but also became increasingly more tachypneic, RR 30-40's crackles went from bases bilaterally to [**2-14**] way up bilaterally.Lasix given 10mg x 2 during the evening without much change in her respiratory status.10units iv insulin given, 1 amps of 50%dextrose, 1amp of sodium bicard, 2gm calc gluconate.LOPRESSOR 7MG IV Q6H, ENALAPRIL 0.625MG IV Q6H TOLERATED.
ID: Continues to receive flagyl, linazolid, pipercillin, ambisome, gent, and acyclovir, the acyclovir was changed from PO to IV-to be given after dialysis.Meds- Lipitor, procardia, synthroid, lisinopril, pepcid, actonel, calcium, MVI Social- Denies tobacco and drugs.'''

# COMMAND ----------

get_sentences_sddl(text_1)  

# COMMAND ----------

get_sentences_spacy(text_1) 

# COMMAND ----------

text_2 = '''ST 109-120 ST. Pt had two 10 beat runs and one 9 beat run of SVT s/p PICC line placement. Stable BP, VT nonsustained. Pt denies CP/SOB. EKG and echo obtained. Cyclying enzymes first CK 69. Cardiology consulted, awaiting echo report. Pt to be started on beta blocker for treatment of NSVT. ? secondary to severe illness.K+ 3.4 IV. IVF with 20meq KCL at 200cc/hr. S/p NSVT pt rec'd 40meq po and 40 meq IV KCL. K+ 3.9 repeat K+ at 8pm. Mg and Ca repleted. Please follow electrolyte SS.
'''

# COMMAND ----------

get_sentences_sddl(text_2)  

# COMMAND ----------

get_sentences_spacy(text_2) 

# COMMAND ----------

text_3 = '''PT. IS A 56 Y/O FEMALE S/P CRANIOTOMY ON 7/16 FOR REMOVAL OF BENIGN CYSTIC LESION. SURGERY PERFORMED AT BIDMC.STARTED ON DILANTIN POST-OP FOR SEIZURE PROPHYLAXIS. 2 DAYS PRIOR TO ADMISSION PT DEVELOPED BILAT. EYE DISCHARGE-- SEEN BY EYE MD AND TREATED WITH SULFATE OPTHALMIC DROPS.ALSO DEVELOPED ORAL SORES AND RASH ON CHEST AND RAPIDLY SPREAD TO TRUNK, ARMS, THIGHS, BUTTOCKS, AND FACE WITHIN 24 HRS.UNABLE TO EAT DUE TO MOUTH PAIN. + FEVER, + DIARRHEA, WEAKNESS. PRESENTED TO EW ON 8/4 WITH TEMP 104.3 SBP 90'S.GIVEN NS FLUID BOLUS, TYLENOL FOR TEMP. SHE PUSTULAR RED RASH ON FACE, RED RASH NOTED ON TRUNK, UPPER EXTREMITIES AND THIGHS. ALSO BOTH EYES DRAINING GREENISH-YELLOW DRAINAGE. ADMITTED TO CCU ( MICU BORDER) FOR CLOSE OBSERVATION.
'''

# COMMAND ----------

get_sentences_sddl(text_3)  

# COMMAND ----------

get_sentences_spacy(text_3) 

# COMMAND ----------

text_4 = ''' Tylenol 650mg po q6h CVS: Aspirin 121.5mg po daily for graft patency, npn 7p-7a: ccu nsg progress note: s/o: does understand and speak some eng, family visiting this eve and states that pt is oriented and appropriate for them resp--ls w/crackles approx 1/2 up, rr 20's, appeared sl sob, on 4l sat when sitting straight up 95-99%, when lying flat or turning s-s sat does drop to 88-93%, pt does not c/o feeling sob, sat does come back up when sitting up, did rec 40mg iv lasix cardiac hr 90's sr w/occ pvc's, bp 95-106/50's, did not c/o any cp during the noc, conts on hep at 600u/hr, ptt during noc 79, am labs pnd, remains off pressors, at this time no further plans to swan pt or for her to go to cath lab, gi--abd soft, non tender to palpation, (+)bs, passing sm amt of brown soft stool, tol po's w/out diff renal--u/o cont'd low during the eve, team decided to give lasix 40mg in setting of crackles, decreased u/o and sob, did diuresis well to lasix, pt approx 700cc neg today access--pt has 3 peripheral iv's in place, all working, unable to draw bloods from pt d/t poor veins, pt is going to need access to draw bloods, central line or picc line social--son & dtr in visiting w/their famlies tonight, pt awake and conversing w/them a/p: cont to monitor/asses cvs follow resp status, additional lasix f/u w/team re: plan of care for her will need iv access for blood draws keep family & pt updated w/plan, Neuro:On propofol gtt dose increased from 20 to 40mcg/kg/min,moves all extrimities to pain,awaken to stimuly easily,purposeful movements.PERL,had an episode of seizure at 1815 <1min when neuro team in for exam ,responded to 2mg ativan on keprra Iv BID.2 grams mag sulfate given, IVF bolus 250 cc started, approx 50 cc in then dc'd d/t PAD's,2.5 mg IV lopressor given x 2 without effect. CCU NURSING 4P-7P S DENIES CP/SOB O. SEE CAREVUE FLOWSHEET FOR COMPLETE VS 1600 O2 SAT 91% ON 5L N/C, 1 U PRBC'S INFUSING, LUNGS CRACKLES BILATERALLY, LASIX 40MG IV ORDERED, 1200 CC U/O W/IMPROVED O2 SATS ON 4L N/C, IABP AT 1:1 BP 81-107/90-117/48-57, HR 70'S SR, GROIN SITES D+I, HEPARIN REMAINS AT 950 U/HR INTEGRELIN AT 2 MCGS/KG A: IMPROVED U/O AFTER LASIX, AWAITING CARDIAC SURGERY P: CONT SUPPORTIVE CARE, REPEAT HCT POST- TRANSFUSION, CHECK LYTES POST-DIURESIS AND REPLACE AS NEEDED, AWAITING CABG-DATE.Given 50mg IV benadryl and 2mg morphine as well as one aspirin PO and inch of NT paste. When pt remained tachycardic w/ frequent ectopy s/p KCL and tylenol for temp - Orders given by Dr. [**Last Name (STitle) 2025**] to increase diltiazem to 120mg PO QID and NS 250ml given X1 w/ moderate effect.Per team, the pts IV sedation was weaned over the course of the day and now infusing @ 110mcg/hr IV Fentanyl & 9mg/hr IV Verced c pt able to open eyes to verbal stimuli and nod head appropriately to simple commands (are you in pain?).  A/P: 73 y/o male remains intubated, IVF boluses x2 for CVP <12, pt continues in NSR on PO amio 400 TID, dopamine gtt weaned down for MAPs>65 but the team felt he is overall receiving about the same amt fentanyl now as he has been in past few days, as the fentanyl patch 100 mcg was added a 48 hrs ago to replace the decrease in the IV fentanyl gtt today (fent patch takes at least 24 hrs to kick in).Started valium 10mg po q6hrs at 1300 with prn IV valium as needed.'''

# COMMAND ----------

get_sentences_sddl(text_4)  

# COMMAND ----------

get_sentences_spacy(text_4) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## *SetenceDetectorDL_HC* Performance and Comparison with *Spacy Sentence Splitter* on *Broken Clinical Texts*

# COMMAND ----------

random_broken_text_1 = '''He was given boluses of MS04 with some effect, he has since been placed on a PCA 
- he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.Repleted with 20 meq kcl po, 30 m
mol K-phos iv and 2 gms mag so4 iv. LASIX CHANGED TO 40 PO BID WHICH IS SAME AS HE TAKES AT HOME - RECEIVED 40 PO IN AM - 700CC U/O TOTAL FOR FLUID NEGATIVE ~ 600 THUS FAR TODAY, ~ 600 NEG LOS pt initially hypertensive 160s-180s gave prn doses if IV hydralazine without effect, labetalol increased from 100 mg to 200 mg PO TID which kept SBP > 160 for rest of night.Transferred to the EW found to be hypertensive BP 253/167 HR 132 ST treated with IV NTG gtt 20-120mcg/ and captopril 12.5mg sl x3. During the day pt's resp status has been very tenuous, responded to lasix in the am but then became hypotensive around 1800 tx with 500cc NS bolus and a unit of RBC did improve her BP to 90-100/60 but also became increasingly more tachypneic, RR 30-40's crackles went from bases bilaterally to [**2-14**] way up bilaterally.Lasix given 10
mg x 2 during the evening without much change in her respiratory status.10units iv insulin given, 1 amps of 50%dextrose, 1amp of sodium bicard, 2gm calc gluconate.LOPRESSOR 7MG IV Q6H, ENALAPRIL 0.625MG IV Q6H TOLERATED.
ID: Continues to receive flagyl, linazolid, pipercillin, ambisome, gent, and acyclovir, the acyclovir was changed from PO to IV-to be given after dialysis.Meds- Lipitor, procardia, synthroid, lisinopril, pepcid, actonel, calcium, MVI Social- Denies tobacco and drugs.'''

# COMMAND ----------

get_sentences_sddl(random_broken_text_1)

# COMMAND ----------

get_sentences_spacy(random_broken_text_1) 

# COMMAND ----------

random_broken_text_2 = '''A 28-year-
old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( 
T2DM ), 
one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 
kg/m2 , presented with a one-
week history of polyuria , polydipsia , poor appetite , and vomiting.Two weeks prior to presentation , she was treated with a five-day course of 
amoxicillin for a respiratory tract infection.She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of 
presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . 
Pertinent laboratory findings on admission were : serum glucose 111 mg
/dl , bicarbonate 18 mmol/l , 
anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg
/dL , total cholesterol 122 
mg/dL , glycated hemoglobin ( HbA1c 
) 10% , and venous pH 7.27 .Serum lipase was normal at 43U/L .Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia.The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission.However , serum chemistry obtained six hours after presentation revealed her glucose was
 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 
 mg/dL , and lipase was 52 U/L.The β-hydroxybutyrate level was obtained and found to be elevated at 5.
 29 
 mmol/L - 
 the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again.The patient was treated with an insulin drip for euDKA 
 and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/
 dL , within 24 hours.Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . 
 The patient was seen by the endocrinology service and she was discharged on 40
  units of insulin glargine at night , 
12 units of insulin lispro with meals , and metformin 1000 mg two times a day.It was determined that all SGLT2 inhibitors should be discontinued indefinitely .'''

# COMMAND ----------

get_sentences_sddl(random_broken_text_2)

# COMMAND ----------

get_sentences_spacy(random_broken_text_2) 

# COMMAND ----------

random_broken_text_3 = ''' Tylenol 650mg po q6h CVS: Aspirin 121.5mg po daily for graft patency.npn 7p-7a: ccu nsg progress note: s/o: does understand and speak some eng, family visiting this eve and states that pt is oriented and appropriate for them resp--ls w/crackles approx 1/2 up, rr 20's, appeared sl sob, on 4l sat when sitting straight up 95-99%, when lying flat or turning s-s sat does drop to 88-93%, pt does not c/o feeling sob, 
sat does come back up when sitting up, did rec 40
mg iv lasix cardiac--hr 90's sr w/occ pvc's, bp 95-
106/50's, did not c/o any cp during the noc, conts on hep at 600u/
hr, ptt during noc 79, am labs pnd, remains off pressors, at this time no further plans to swan pt or for her to go to cath lab, gi--abd soft, non tender to palpation, (+)bs, passing sm amt of brown soft stool, tol po's w/out diff renal--u/o cont'd low during the eve,
 team decided to give lasix 40mg in setting of crackles, decreased u/o 
and sob, did diuresis well to lasix, pt approx 700cc neg today access--pt has 3 peripheral iv's in place, all working, unable to draw bloods from pt d/t poor veins, pt is going to need access to draw bloods, ?central line or picc line social--son & dtr in visiting w/their famlies tonight, pt awake and conversing w/them a/p: cont to monitor/asses cvs follow resp status, additional
 lasix f/u w/team re: plan of care for her will need iv access for blood draws keep family & pt updated w/plan, Neuro:On propofol gtt dose increased from 20 to 40mcg/kg/min,moves all extrimities to pain,awaken to stimuly easily,purposeful movements.PERL,had an episode of seizure at 1815 <1min when neuro team in for exam ,responded to 2mg ativan on keprra Iv BID.2 grams mag sulfate given, IVF bolus 250 cc started, approx 50 cc in then dc'd d/t PAD's, 
 2.5 mg IV lopressor given x 2 without effect. CCU NURSING 4P-7P S DENIES CP/SOB O. SEE CAREVUE FLOWSHEET FOR COMPLETE VS 1600 O2 SAT 91% ON 5L N/C, 1 U PRBC'S INFUSING, LUNGS CRACKLES BILATERALLY, LASIX 40MG IV ORDERED, 1200 CC U/O W/IMPROVED O2 SATS ON 4L N/C, IABP AT 1:1 BP 81-107/90-117/48-57, HR 70'S SR, GROIN SITES D+I, HEPARIN REMAINS AT 950 U/HR INTEGRELIN AT 2 MCGS/KG A: IMPROVED U/O AFTER LASIX, AWAITING CARDIAC SURGERY P: CONT SUPPORTIVE CARE, 
 REPEAT HCT POST-TRANSFUSION, CHECK LYTES POST-DIURESIS AND REPLACE AS NEEDED, AWAITING CABG -DATE.Given 50mg IV benadryl and 2mg morphine as well as one aspirin PO and inch of NT paste. When pt remained tachycardic w/ frequent ectopy s/p KCL and tylenol for temp - Orders given by Dr. [**Last Name (STitle) 2025**] to increase diltiazem to 120mg PO QID and NS 250ml given X1 w/ moderate effect.Per team, the pts IV sedation was weaned over the course of the 
 day and now infusing @ 110mcg/hr IV Fentanyl & 9mg/hr IV Verced c pt able to open eyes to verbal stimuli and nod head appropriately to simple commands (are you in pain?) .  A/P: 73 y/o male remains intubated, IVF boluses x2 for CVP <12, pt continues in NSR on PO amio 400 TID, dopamine gtt weaned down for MAPs>65 but the team felt he is overall receiving about the same amt fentanyl now as he has been in past few days, as the fentanyl patch 100
mcg was added a 48 hrs ago to replace the decrease in the IV fentanyl gtt today (fent patch takes at least 24 hrs to kick in).Started valium 10mg po q6hrs at 1300 with prn IV valium as needed.'''

# COMMAND ----------

get_sentences_sddl(random_broken_text_3)

# COMMAND ----------

get_sentences_spacy(random_broken_text_3)

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #