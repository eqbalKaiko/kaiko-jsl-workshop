# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced Image Processing and Text Recognition with Spark OCR

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

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)


print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image (or Natural Scene) to Text

# COMMAND ----------

import base64
import sparkocr
from sparkocr.enums import *
from sparkocr.metrics import score
from sparkocr.transformers import *
from sparkocr.utils import display_image, to_pil_image
from sparkocr.databricks import display_images
from pyspark.sql import functions as F
import matplotlib.pyplot as plt

spark.sql("set spark.sql.legacy.allowUntypedScalaUDF=true")

sys.path.append('/databricks/driver/')
sys.path.append('/databricks/driver/display_images.py')
sys.path.append('/databricks/driver/score.py')

# COMMAND ----------

# MAGIC %md
# MAGIC # Pdf to Text

# COMMAND ----------

!wget -q -O sample_doc.pdf http://www.asx.com.au/asxpdf/20171103/pdf/43nyyw9r820c6r.pdf
dbutils.fs.cp("file:/databricks/driver/sample_doc.pdf", "dbfs:/")

# COMMAND ----------

def pipeline():
    
    # Transforrm PDF document to images per page
    pdf_to_image = PdfToImage()\
          .setInputCol("content")\
          .setOutputCol("image")

    # Run OCR
    ocr = ImageToText()\
          .setInputCol("image")\
          .setOutputCol("text")\
          .setConfidenceThreshold(65)
    
    pipeline = PipelineModel(stages=[
        pdf_to_image,
        ocr
    ])
    
    return pipeline

# COMMAND ----------

pdf = '/sample_doc.pdf'
pdf_example_df = spark.read.format("binaryFile").load(pdf).cache()

# COMMAND ----------

result = pipeline().transform(pdf_example_df).cache()

# COMMAND ----------

result.select("pagenum","text", "confidence").show()

# COMMAND ----------

result.select("text").collect()

# COMMAND ----------

print("\n".join([row.text for row in result.select("text").collect()]))


# COMMAND ----------

# MAGIC %md
# MAGIC ##  With Skew Correction

# COMMAND ----------

from sparkocr.transformers import *
from pyspark.ml import PipelineModel
from sparkocr.utils import display_image
from sparkocr.metrics import score

# COMMAND ----------

def ocr_pipeline(skew_correction=False):
    
    # Transforrm PDF document to images per page
    pdf_to_image = PdfToImage()\
          .setInputCol("content")\
          .setOutputCol("image")

    # Image skew corrector 
    skew_corrector = ImageSkewCorrector()\
          .setInputCol("image")\
          .setOutputCol("corrected_image")\
          .setAutomaticSkewCorrection(skew_correction)

    # Run OCR
    ocr = ImageToText()\
          .setInputCol("corrected_image")\
          .setOutputCol("text")
    
    pipeline = PipelineModel(stages=[
        pdf_to_image,
        skew_corrector,
        ocr
    ])
    
    return pipeline

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ocr/400_rot.pdf
dbutils.fs.cp("file:/databricks/driver/400_rot.pdf", "dbfs:/")

# COMMAND ----------

pdf_rotated_df = spark.read.format("binaryFile").load('/400_rot.pdf').cache()

# COMMAND ----------

pdf_pipeline = ocr_pipeline(False) 

result = pdf_pipeline.transform(pdf_rotated_df).cache()


# COMMAND ----------

result.show()

# COMMAND ----------

result.select("pagenum").collect()[0].pagenum

# COMMAND ----------

img = result.select("image").collect()[0].image
img_pil = to_pil_image(img, img.mode)

plt.figure(figsize=(30,20))
plt.imshow(img_pil, cmap='gray')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display recognized text without skew correction

# COMMAND ----------

result.select("pagenum","text", "confidence").show()


# COMMAND ----------

print("\n".join([row.text for row in result.select("text").collect()]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display results with skew correction

# COMMAND ----------

pdf_pipeline = ocr_pipeline(True) 

corrected_result = pdf_pipeline.transform(pdf_rotated_df).cache()

print("\n".join([row.text for row in corrected_result.select("text").collect()]))


# COMMAND ----------

corrected_result.select("pagenum","text", "confidence").show(truncate=50)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Display skew corrected images

# COMMAND ----------

img = corrected_result.select("corrected_image").collect()[0].corrected_image
img_pil = to_pil_image(img, img.mode)

plt.figure(figsize=(30,20))
plt.imshow(img_pil, cmap='gray')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute score and compare
# MAGIC Read original text and calculate scores for both results.

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ocr/400.txt
dbutils.fs.cp("file:/databricks/driver/400.txt", "dbfs:/")

# COMMAND ----------

detected = "\n".join([row.text for row in result.collect()])
corrected_detected = "\n".join([row.text for row in corrected_result.collect()])

# read original text
pdf_rotated_text = open('400.txt', "r").read()

# compute scores
detected_score = score(pdf_rotated_text, detected)
corrected_score = score(pdf_rotated_text, corrected_detected)

#  print scores
print("Score without skew correction: {0}".format(detected_score))
print("Score with skew correction: {0}".format(corrected_score))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading multiple pdfs from folder

# COMMAND ----------

pdf_path = "/*.pdf"

pdfs = spark.read.format("binaryFile").load(pdf_path).cache()
#images = spark.read.format("binaryFile").load('text_with_noise.png').cache()

pdfs.count()

# COMMAND ----------

# Transforrm PDF document to images per page
pdf_to_image = PdfToImage()\
      .setInputCol("content")\
      .setOutputCol("image")

# Run OCR
ocr = ImageToText()\
      .setInputCol("image")\
      .setOutputCol("text")\
      .setConfidenceThreshold(65)\
      .setIgnoreResolution(False)

ocr_pipeline = PipelineModel(stages=[
    pdf_to_image,
    ocr
])


# COMMAND ----------

results = ocr_pipeline.transform(pdfs)


# COMMAND ----------

results.columns

# COMMAND ----------

results.select('path','confidence','text').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image processing after reading a pdf

# COMMAND ----------

from sparkocr.enums import *

# Read binary as image
pdf_to_image = PdfToImage()\
  .setInputCol("content")\
  .setOutputCol("image")\
  .setResolution(400)

# Binarize using adaptive tresholding
binarizer = ImageAdaptiveThresholding()\
  .setInputCol("image")\
  .setOutputCol("binarized_image")\
  .setBlockSize(91)\
  .setOffset(50)

# Apply morphology opening
opening = ImageMorphologyOperation()\
  .setKernelShape(KernelShape.SQUARE)\
  .setOperation(MorphologyOperationType.OPENING)\
  .setKernelSize(3)\
  .setInputCol("binarized_image")\
  .setOutputCol("opening_image")

# Remove small objects
remove_objects = ImageRemoveObjects()\
  .setInputCol("opening_image")\
  .setOutputCol("corrected_image")\
  .setMinSizeObject(130)

# Image Layout Analyzer for detect regions
image_layout_analyzer = ImageLayoutAnalyzer()\
  .setInputCol("corrected_image")\
  .setOutputCol("region")\

draw_regions = ImageDrawRegions()\
  .setInputCol("corrected_image")\
  .setInputRegionsCol("region")\
  .setOutputCol("image_with_regions")

# Run tesseract OCR for corrected image
ocr_corrected = ImageToText()\
  .setInputCol("corrected_image")\
  .setOutputCol("corrected_text")\
  .setPositionsCol("corrected_positions")\
  .setConfidenceThreshold(65)

# Run OCR for original image
ocr = ImageToText()\
  .setInputCol("image")\
  .setOutputCol("text")

# OCR pipeline
image_pipeline = PipelineModel(stages=[
    pdf_to_image,
    binarizer,
    opening,
    remove_objects,
    image_layout_analyzer,
    draw_regions,
    ocr,
    ocr_corrected
])

# COMMAND ----------

! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/pdfs/noised.pdf
dbutils.fs.cp("file:/databricks/driver/noised.pdf", "dbfs:/")

# COMMAND ----------

image_df = spark.read.format("binaryFile").load('/noised.pdf').cache()
image_df.show()

# COMMAND ----------

result = image_pipeline.transform(image_df).cache()

# COMMAND ----------

for r in result.distinct().collect():
    
    print("Original: %s" % r.path)
    img = r.image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(30,20))
    plt.imshow(img_pil, cmap='gray')
    
    
    print("Corrected: %s" % r.path)
    img = r.corrected_image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(30,20))
    plt.imshow(img_pil, cmap='gray')
    
    
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results with original image

# COMMAND ----------

grouped_results = result.groupBy("path", "pagenum").agg(F.concat_ws("", F.collect_list("text")).alias("text"))
for row in grouped_results.collect():
    print("Filename:\n%s , page: %d" % (row.path, row.pagenum))
    print("Recognized text:\n%s" % row.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results with corrected image

# COMMAND ----------

grouped_results = result.groupBy("path", "pagenum").agg(F.concat_ws("", F.collect_list("corrected_text")).alias("corrected_text"))
for row in grouped_results.collect():
    print("Filename:\n%s , page: %d" % (row.path, row.pagenum))
    print("Recognized text:\n%s" % row.corrected_text)

# COMMAND ----------

result.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abby output

# COMMAND ----------

abbyy = """-----
% Date: 7/16/68
X*: I; * • ■ Sample No. 5031___ — .*
•* Original request made by _____Mr. C. L. Tucker, Jr. on
Sample specifications written by
BLEND CASING RECASING
OLD GOLD STRAIGHT Tobacco Blend
Control for Sample No. 5030
John H. M. Bohlken
FINAL FLAVOR
) 7/10/68
MENTHOL FLAVOR
• Cigarettes; * . .v\ . /,*, *, S •
Brand --------- OLD GOLD STRAIGHT -V . ••••
; . L e n g t h ------- — 85 mm. . : '
Circumference-- 25.3 mm. • ' *;. • •
P a p e r ---------- Ecusta 556 • * .
F i r m n e s s---- —— OLD GOLD STRAIGHT . ! •■'
D r a w ___________ OLD GOLD STRAIGHT
W e i g h t --------- 0LD GOLD STRAIGHT Wrappings: « -
Tipping Paper — — *
p H n f —. — — _ _ ~ L a b e l s ----OLD GOLD STRAIGHT
( • Filter Length-- . — Closures--- Standard Blue .
^ ^ ; • Tear Tape— Gold
Cartons --- OLD GOLD STRAIGHT
s Requirements: . - •' • Markings-- Sample number on each
• pack and carton Laboratory----- One Tray .
O t h e r s --------- * , s • • . 4
Laboratory A n a l ysis^ I " '/***• * 7 ' ^ ^
Tars and Nicotine, Taste Panel, Burning Time, Gas Phase Analysis,
Benzo (A) Pyrene Analyses — J-ZZ-Zf'- (£. / •
Responsibility;
Tobacco B l e n d ------Manufacturing - A. Kraus . . * -
Filter Production--- —
• Making & P a c k i n g---Product Development , John H. M. Bohlken
Shipping -----------
Reports:
t
Written by — John H. M. Bohlken
Original to - Mr. C. L. Tucker, Jr.
Copies t o ---Dr. A. W. Spears
• 9 ..
"""

# COMMAND ----------

for r in result.select("path","image","image_with_regions").distinct().collect():
    
    print("Original: %s" % r.path)
    img = r.image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(30,20))
    plt.imshow(img_pil, cmap='gray')
    
    
    print("Corrected: %s" % r.path)
    img = r.image_with_regions
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(30,20))
    plt.imshow(img_pil, cmap='gray')
    
    
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Image (or Natural Scene) to Text

# COMMAND ----------

! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/images/text_with_noise.png
dbutils.fs.cp("file:/databricks/driver/text_with_noise.png", "dbfs:/")

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ocr/natural_scene.jpeg
dbutils.fs.cp("file:/databricks/driver/natural_scene.jpeg", "dbfs:/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text from Scene

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read DOCX document as binary file

# COMMAND ----------

# MAGIC %md
# MAGIC # DOCX Processing (version 1.10.0)

# COMMAND ----------

image_df = spark.read.format("binaryFile").load('/text_with_noise.png').cache()

# Read binary as image
binary_to_image = BinaryToImage()
binary_to_image.setInputCol("content")
binary_to_image.setOutputCol("image")

# Scale image
scaler = ImageScaler()
scaler.setInputCol("image")
scaler.setOutputCol("scaled_image")
scaler.setScaleFactor(2.0)

# Binarize using adaptive tresholding
binarizer = ImageAdaptiveThresholding()
binarizer.setInputCol("scaled_image")
binarizer.setOutputCol("binarized_image")
binarizer.setBlockSize(71)
binarizer.setOffset(65)

remove_objects = ImageRemoveObjects()
remove_objects.setInputCol("binarized_image")
remove_objects.setOutputCol("cleared_image")
remove_objects.setMinSizeObject(400)
remove_objects.setMaxSizeObject(4000)

# Run OCR
ocr = ImageToText()
ocr.setInputCol("cleared_image")
ocr.setOutputCol("text")
ocr.setConfidenceThreshold(50)
ocr.setIgnoreResolution(False)

# OCR pipeline
noisy_pipeline = PipelineModel(stages=[
    binary_to_image,
    scaler,
    binarizer,
    remove_objects,
    ocr
])


result = noisy_pipeline \
.transform(image_df) \
.cache()


for r in result.distinct().collect():
  
    print("Original: %s" % r.path)
    img = r.image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(30,20))
    plt.imshow(img_pil, cmap='gray')
    
    print("Binarized")
    img = r.binarized_image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(30,20))
    plt.imshow(img_pil, cmap='gray')
  
    print("Removing objects")
    img = r.cleared_image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(30,20))
    plt.imshow(img_pil, cmap='gray')
    
    
    plt.show()

# COMMAND ----------

print("\n".join([row.text for row in result.select("text").collect()]))

# COMMAND ----------

doc_example_df.show()

# COMMAND ----------

doc_example_df = spark.read.format("binaryFile").load("file:/databricks/python/lib/python3.7/site-packages/sparkocr/resources/ocr/docs/doc2.docx").cache()

# COMMAND ----------

import pkg_resources
pkg_resources.resource_filename('sparkocr', 'resources/ocr/docs/doc2.docx')

# COMMAND ----------

# MAGIC %md
# MAGIC ## DocxtoText

# COMMAND ----------

image_df = spark.read.format("binaryFile").load('/natural_scene.jpeg').cache()

# Apply morphology opening
morpholy_operation = ImageMorphologyOperation()
morpholy_operation.setKernelShape(KernelShape.DISK)
morpholy_operation.setKernelSize(5)
morpholy_operation.setOperation("closing")
morpholy_operation.setInputCol("cleared_image")
morpholy_operation.setOutputCol("corrected_image")

# Run OCR
ocr = ImageToText()
ocr.setInputCol("corrected_image")
ocr.setOutputCol("text")
ocr.setConfidenceThreshold(50)
ocr.setIgnoreResolution(False)

# OCR pipeline
scene_pipeline = PipelineModel(stages=[
    binary_to_image,
    scaler,
    binarizer,
    remove_objects,
    morpholy_operation,
    ocr
])

result = scene_pipeline \
.transform(image_df) \
.cache()


for r in result.distinct().collect():
      
    print("Original: %s" % r.path)
    img = r.image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(20,12))
    plt.imshow(img_pil, cmap='gray')
    
    print("Binarized")
    img = r.binarized_image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(20,12))
    plt.imshow(img_pil, cmap='gray')
  
    print("Removing objects")
    img = r.cleared_image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(20,12))
    plt.imshow(img_pil, cmap='gray')
    
    print("Morphology closing")
    img = r.corrected_image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(20,12))
    plt.imshow(img_pil, cmap='gray')
    
    
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract text using DocToText transformer

# COMMAND ----------

from sparkocr.transformers import *

doc_to_text = DocToText()
doc_to_text.setInputCol("content")
doc_to_text.setOutputCol("text")

result = doc_to_text.transform(doc_example_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display result DataFrame

# COMMAND ----------

result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display extracted text

# COMMAND ----------

print("\n".join([row.text for row in result.select("text").collect()]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## DocxToTextTable
# MAGIC ### (Extracting table data from Microsoft DOCX documents)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preview document using DocToPdf and PdfToImage transformers

# COMMAND ----------

for r in image_df.select("image").collect():
    display_image(r.image)

# COMMAND ----------

image_df = PdfToImage().transform(DocToPdf().setOutputCol("content").transform(doc_example_df))

for r in image_df.distinct().collect():
      
    print("Original: %s" % r.path)
    img = r.image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(24,16))
    plt.imshow(img_pil, cmap='gray')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract text using DocToText transformer

# COMMAND ----------

doc_to_table = DocToTextTable()
doc_to_table.setInputCol("content")
doc_to_table.setOutputCol("tables")

result = doc_to_table.transform(doc_example_df)

result.show()

# COMMAND ----------

result.select(result["tables.chunks"].getItem(3)["chunkText"]).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display extracted data in JSON format

# COMMAND ----------

import json
df_json = result.select("tables").toJSON()
for row in df_json.collect():
    print(json.dumps(json.loads(row), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC # Text to Pdf

# COMMAND ----------

def pipeline():
    # Transforrm PDF document to images per page
    pdf_to_image = PdfToImage() \
        .setInputCol("content") \
        .setOutputCol("image") \
        .setKeepInput(True)
    
    # Run OCR
    ocr = ImageToText() \
        .setInputCol("image") \
        .setOutputCol("text") \
        .setConfidenceThreshold(60) \
        .setIgnoreResolution(False) \
        .setPageSegMode(PageSegmentationMode.SPARSE_TEXT)
    
    # Render results to PDF
    textToPdf = TextToPdf() \
        .setInputCol("positions") \
        .setInputImage("image") \
        .setOutputCol("pdf")

    pipeline = PipelineModel(stages=[
        pdf_to_image,
        ocr,
        textToPdf
    ])
    
    return pipeline

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/ocr/test_document.pdf
dbutils.fs.cp("file:/databricks/driver/test_document.pdf", "dbfs:/")

# COMMAND ----------

pdf_example_df = spark.read.format("binaryFile").load('/test_document.pdf').cache()


# COMMAND ----------

result = pipeline().transform(pdf_example_df).cache()

# COMMAND ----------

result.columns

# COMMAND ----------

image_df = PdfToImage().transform(pdf_example_df).select("image").collect()[0].image

img_pil = to_pil_image(image_df, image_df.mode)

plt.figure(figsize=(24,16))
plt.imshow(img_pil, cmap='gray')

# COMMAND ----------

# Store results to pdf file
pdf = result.select("pdf").head().pdf

pdfFile = open("result.pdf", "wb")

pdfFile.write(pdf)

pdfFile.close()

# COMMAND ----------

# Convert pdf to image and display¶

image_df = PdfToImage() \
    .setInputCol("pdf") \
    .setOutputCol("image") \
    .transform(result.select("pdf", "path"))

    
for r in image_df.distinct().collect():
      
    print("Original: %s" % r.path)
    img = r.image
    img_pil = to_pil_image(img, img.mode)

    plt.figure(figsize=(24,16))
    plt.imshow(img_pil, cmap='gray')


# COMMAND ----------

# MAGIC %md
# MAGIC # DICOM Image Deidentifier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deidentification Pipeline

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl

# COMMAND ----------

def deidentification_nlp_pipeline(input_column, prefix = ""):
    document_assembler = DocumentAssembler() \
        .setInputCol(input_column) \
        .setOutputCol(prefix + "document")

    # Sentence Detector annotator, processes various sentences per line
    sentence_detector = SentenceDetector() \
        .setInputCols([prefix + "document"]) \
        .setOutputCol(prefix + "sentence")

    tokenizer = Tokenizer() \
        .setInputCols([prefix + "sentence"]) \
        .setOutputCol(prefix + "token")

    # Clinical word embeddings
    word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
        .setInputCols([prefix + "sentence", prefix + "token"]) \
        .setOutputCol(prefix + "embeddings")
    # NER model trained on i2b2 (sampled from MIMIC) dataset
    clinical_ner = MedicalNerModel.pretrained("ner_deid_large", "en", "clinical/models") \
        .setInputCols([prefix + "sentence", prefix + "token", prefix + "embeddings"]) \
        .setOutputCol(prefix + "ner")

    custom_ner_converter = NerConverter() \
        .setInputCols([prefix + "sentence", prefix + "token", prefix + "ner"]) \
        .setOutputCol(prefix + "ner_chunk") \
        .setWhiteList(['NAME', 'AGE', 'CONTACT', 'LOCATION', 'PROFESSION', 'PERSON', 'DATE'])

    nlp_pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            word_embeddings,
            clinical_ner,
            custom_ner_converter
        ])
    empty_data = spark.createDataFrame([[""]]).toDF(input_column)
    nlp_model = nlp_pipeline.fit(empty_data)
    return nlp_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define OCR transformers and pipeline

# COMMAND ----------

# Convert to images
binary_to_image = BinaryToImage() \
    .setInputCol("content") \
    .setOutputCol("image_raw")

# Extract text from image
ocr = ImageToText() \
    .setInputCol("image_raw") \
    .setOutputCol("text") \
    .setIgnoreResolution(False) \
    .setPageIteratorLevel(PageIteratorLevel.SYMBOL) \
    .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
    .setConfidenceThreshold(70)

# Found coordinates of sensitive data
position_finder = PositionFinder() \
    .setInputCols("ner_chunk") \
    .setOutputCol("coordinates") \
    .setPageMatrixCol("positions") \
    .setMatchingWindow(1000) \
    .setPadding(1)

# Draw filled rectangle for hide sensitive data
drawRegions = ImageDrawRegions()  \
    .setInputCol("image_raw")  \
    .setInputRegionsCol("coordinates")  \
    .setOutputCol("image_with_regions")  \
    .setFilledRect(True) \
    .setRectColor(Color.gray)
    

# OCR pipeline
pipeline = Pipeline(stages=[
    binary_to_image,
    ocr,
    deidentification_nlp_pipeline(input_column="text"),
    position_finder,
    drawRegions
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Image

# COMMAND ----------

import pkg_resources
pkg_resources.resource_filename('sparkocr', 'resources/ocr/images/p1.jpg')

# COMMAND ----------

image_df = spark.read.format("binaryFile").load("file:/databricks/python/lib/python3.7/site-packages/sparkocr/resources/ocr/images/p1.jpg")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Pipeline

# COMMAND ----------

result = pipeline.fit(image_df).transform(image_df).cache()

# COMMAND ----------

result.select('text').show(1, truncate=False)

# COMMAND ----------

# Chunks to hide
result.select('ner_chunk').show(2, False)

# COMMAND ----------

# Coordinates of Chunks to Hide
result.select('coordinates').show(2, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show original and deidentified images

# COMMAND ----------

for r in result.select("image_raw", "image_with_regions").collect():
    img_orig = r.image_raw
    img_deid = r.image_with_regions

    img_pil_orig = to_pil_image(img_orig, img_orig.mode)
    img_pil_deid = to_pil_image(img_deid, img_deid.mode)

    plt.figure(figsize=(24,16))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil_orig, cmap='gray')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(img_pil_deid, cmap='gray')
    plt.title("de-id'd")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Dicom to Image

# COMMAND ----------

! mkdir dicom
! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/dicom/deidentify-brains-front-medical-3.dcm -O /databricks/driver/dicom/dicom_1.dcm
! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/dicom/deidentify-medical-1.dcm  -O /databricks/driver/dicom/dicom_2.dcm
! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-ocr-workshop/master/jupyter/data/dicom/deidentify-medical-2.dcm  -O /databricks/driver/dicom/dicom_3.dcm
  
dbutils.fs.cp("file:/databricks/driver/dicom/dicom_1.dcm", "dbfs:/", True)  
dbutils.fs.cp("file:/databricks/driver/dicom/dicom_2.dcm", "dbfs:/", True)  
dbutils.fs.cp("file:/databricks/driver/dicom/dicom_3.dcm", "dbfs:/", True)  


# COMMAND ----------

dicom_path = 'dbfs:/*.dcm'

# Read dicom file as binary file
dicom_df = spark.read.format("binaryFile").load(dicom_path)

# COMMAND ----------

dicomToImage = DicomToImage() \
  .setInputCol("content") \
  .setOutputCol("image") \
  .setMetadataCol("meta")

data = dicomToImage.transform(dicom_df)

for r in data.collect():
    img = r.image
    img_pil = to_pil_image(img, img.mode)
    plt.figure(figsize=(24,16))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil, cmap='gray')

# COMMAND ----------

# Extract text from image
ocr = ImageToText() \
    .setInputCol("image") \
    .setOutputCol("text") \
    .setIgnoreResolution(False) \
    .setOcrParams(["preserve_interword_spaces=0"])


print("\n".join([row.text for row in ocr.transform(data).select("text").collect()]))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Deidentify PHI on DICOM

# COMMAND ----------

# MAGIC %md
# MAGIC https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrDicomDeIdentification.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC ## More example here
# MAGIC 
# MAGIC https://github.com/JohnSnowLabs/spark-ocr-workshop