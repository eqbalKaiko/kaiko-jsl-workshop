# Databricks notebook source
# MAGIC %md
# MAGIC # Extract text from scanned documents with Spark OCR
# MAGIC 
# MAGIC This notebook will illustrates how to:
# MAGIC * Load example PDF
# MAGIC * Preview it
# MAGIC * Recognize text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import OCR transformers and utils

# COMMAND ----------

from sparkocr.transformers import *
from sparkocr.databricks import display_images
from pyspark.ml import PipelineModel

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define OCR transformers and pipeline
# MAGIC * Transforrm binary data to Image schema using [BinaryToImage](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#binarytoimage). More details about Image Schema [here](https://nlp.johnsnowlabs.com/docs/en/ocr_structures#image-schema).
# MAGIC * Recognize text using [ImageToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#imagetotext) transformer.

# COMMAND ----------

def pipeline():
    
    # Transforrm PDF document to struct image format
    pdf_to_image = PdfToImage()
    pdf_to_image.setInputCol("content")
    pdf_to_image.setOutputCol("image")
    pdf_to_image.setResolution(200)
    pdf_to_image.setPartitionNum(8)

    # Run OCR
    ocr = ImageToText()
    ocr.setInputCol("image")
    ocr.setOutputCol("text")
    ocr.setConfidenceThreshold(65)
    
    pipeline = PipelineModel(stages=[
        pdf_to_image,
        ocr
    ])
    
    return pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Copy example files from OCR resources to DBFS

# COMMAND ----------

import pkg_resources
import shutil, os
ocr_examples = "/dbfs/FileStore/examples"
resources = pkg_resources.resource_filename('sparkocr', 'resources')
if not os.path.exists(ocr_examples):
  shutil.copytree(resources, ocr_examples)

# COMMAND ----------

# MAGIC %fs ls /FileStore/examples/ocr/pdfs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read PDF document as binary file from DBFS

# COMMAND ----------

pdf_example = '/FileStore/examples/ocr/pdfs/test_document.pdf'
pdf_example_df = spark.read.format("binaryFile").load(pdf_example).cache()
display(pdf_example_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview PDF using _display_images_ function

# COMMAND ----------

display_images(PdfToImage().setOutputCol("image").transform(pdf_example_df), limit=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run OCR pipelines

# COMMAND ----------

result = pipeline().transform(pdf_example_df).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display results

# COMMAND ----------

display(result.select("pagenum", "text", "confidence"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clear cache

# COMMAND ----------

result.unpersist()
pdf_example_df.unpersist()