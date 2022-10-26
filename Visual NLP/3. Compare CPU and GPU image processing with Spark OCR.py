# Databricks notebook source
# MAGIC %md
# MAGIC # Compare CPU and GPU image preprocessing in Spark OCR
# MAGIC * Load images from S3
# MAGIC * Preview images
# MAGIC * Recognize text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import OCR transformers and utils

# COMMAND ----------

from sparkocr.transformers import *
from sparkocr.enums import *
from sparkocr.databricks import display_images
from pyspark.ml import PipelineModel

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define OCR transformers and pipeline
# MAGIC * Transforrm binary data to Image schema using [BinaryToImage](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#binarytoimage). More details about Image Schema [here](https://nlp.johnsnowlabs.com/docs/en/ocr_structures#image-schema).
# MAGIC * Recognize text using [ImageToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#imagetotext) transformer.

# COMMAND ----------

def pipeline():
    
    # Transforrm binary data to struct image format
    binary_to_image = BinaryToImage()
    binary_to_image.setInputCol("content")
    binary_to_image.setOutputCol("image")

    # Run OCR
    ocr = ImageToText()
    ocr.setInputCol("image")
    ocr.setOutputCol("text")
    ocr.setConfidenceThreshold(65)
    
    pipeline = PipelineModel(stages=[
        binary_to_image,
        ocr
    ])
    
    return pipeline

# COMMAND ----------

def pipelineCPU():
    
    # Transforrm binary data to struct image format
    binary_to_image = BinaryToImage()
    binary_to_image.setInputCol("content")
    binary_to_image.setOutputCol("image")
    
    # Scale image
    scaler = ImageScaler()\
      .setInputCol("image")\
      .setOutputCol("scaled_image")\
      .setScaleFactor(2.0)
    
    # Binaraze image
    adaptive_thresholding = ImageAdaptiveBinarizer() \
      .setInputCol("scaled_image") \
      .setOutputCol("corrected_image") \
      .setMethod(ThresholdingMethod.OTSU)
    
    # Apply erosion
    erosion = ImageMorphologyOperation() \
      .setKernelShape(KernelShape.SQUARE) \
      .setKernelSize(1) \
      .setOperation("erosion") \
      .setInputCol("corrected_image") \
      .setOutputCol("eroded_image")

    # Run OCR
    ocr = ImageToText()
    ocr.setInputCol("eroded_image")
    ocr.setOutputCol("text")
    ocr.setConfidenceThreshold(65)
    
    pipeline = PipelineModel(stages=[
        binary_to_image,
        scaler,
        adaptive_thresholding,
        erosion,
        ocr
    ])
    
    return pipeline

# COMMAND ----------

def pipelineGPU():
    
    # Transforrm binary data to struct image format
    binary_to_image = BinaryToImage()
    binary_to_image.setInputCol("content")
    binary_to_image.setOutputCol("image")
    
    # Image transformation on GPU
    transformer = GPUImageTransformer() \
      .addScalingTransform(2) \
      .addOtsuTransform() \
      .addErodeTransform(1, 1) \
      .setInputCol("image") \
      .setOutputCol("eroded_image") \

    # Run OCR
    ocr = ImageToText()
    ocr.setInputCol("eroded_image")
    ocr.setOutputCol("text")
    ocr.setConfidenceThreshold(65)
    
    pipeline = PipelineModel(stages=[
        binary_to_image,
        transformer,
        ocr
    ])
    
    return pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download images from public S3 bucket to DBFS

# COMMAND ----------

# MAGIC %sh
# MAGIC OCR_DIR=/dbfs/tmp/ocr_1
# MAGIC if [ ! -d "$OCR_DIR" ]; then
# MAGIC     mkdir $OCR_DIR
# MAGIC     cd $OCR_DIR
# MAGIC     wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/ocr/datasets/news.2B.0.png.zip
# MAGIC     unzip news.2B.0.png.zip
# MAGIC fi

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/ocr_1/0/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read images as binary files
# MAGIC from DBFS

# COMMAND ----------

images_path = "/tmp/ocr_1/0/*.png"
images_example_df = spark.read.format("binaryFile").load(images_path).cache()
display(images_example_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data from s3 directly using credentials

# COMMAND ----------

# ACCESS_KEY = ""
# SECRET_KEY = ""
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", ACCESS_KEY)
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", SECRET_KEY)
# imagesPath = "s3a://dev.johnsnowlabs.com/ocr/datasets/news.2B/0/*.tif"
# imagesExampleDf = spark.read.format("binaryFile").load(imagesPath).cache()
# display(imagesExampleDf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display count of images

# COMMAND ----------

images_example_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview images using _display_images_ function

# COMMAND ----------

display_images(BinaryToImage().transform(images_example_df), limit=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run OCR pipeline without preprocessing

# COMMAND ----------

result = pipeline().transform(images_example_df.repartition(8)).cache()
result.count()

# COMMAND ----------

display(result.select("text", "confidence"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## CPU image preprocessing

# COMMAND ----------

resultCPU = pipelineCPU().transform(images_example_df.repartition(8)).cache()
resultCPU.count()

# COMMAND ----------

display(resultCPU.select("text", "confidence"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPU image preprocessing

# COMMAND ----------

resultGPU = pipelineGPU().transform(images_example_df.repartition(8)).cache()
resultGPU.count()

# COMMAND ----------

display(resultGPU.select("text", "confidence"))