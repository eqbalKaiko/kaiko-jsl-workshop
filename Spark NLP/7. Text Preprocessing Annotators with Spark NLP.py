# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Text Preprocessing Annotators with Spark NLP

# COMMAND ----------

import sparknlp
import pandas as pd

print("Spark NLP version", sparknlp.version())

print("Apache Spark version:", spark.version)

spark

# COMMAND ----------

from sparknlp.base import *
from sparknlp.annotator import *

# COMMAND ----------

# MAGIC %md
# MAGIC **Note** Read this article if you want to understand the basic concepts in Spark NLP.
# MAGIC 
# MAGIC https://towardsdatascience.com/introduction-to-spark-nlp-foundations-and-basic-components-part-i-c83b7629ed59

# COMMAND ----------

# MAGIC %md
# MAGIC ## Annotators and Transformer Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC In Spark NLP, all Annotators are either Estimators or Transformers as we see in Spark ML. An Estimator in Spark ML is an algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model. A Transformer is an algorithm which can transform one DataFrame into another DataFrame. E.g., an ML model is a Transformer that transforms a DataFrame with features into a DataFrame with predictions.
# MAGIC In Spark NLP, there are two types of annotators: AnnotatorApproach and AnnotatorModel
# MAGIC AnnotatorApproach extends Estimators from Spark ML, which are meant to be trained through fit(), and AnnotatorModel extends Transformers which are meant to transform data frames through transform().
# MAGIC Some of Spark NLP annotators have a Model suffix and some do not. The model suffix is explicitly stated when the annotator is the result of a training process. Some annotators, such as Tokenizer are transformers but do not contain the suffix Model since they are not trained, annotators. Model annotators have a pre-trained() on its static object, to retrieve the public pre-trained version of a model.
# MAGIC Long story short, if it trains on a DataFrame and produces a model, it’s an AnnotatorApproach; and if it transforms one DataFrame into another DataFrame through some models, it’s an AnnotatorModel (e.g. WordEmbeddingsModel) and it doesn’t take Model suffix if it doesn’t rely on a pre-trained annotator while transforming a DataFrame (e.g. Tokenizer).

# COMMAND ----------

!wget -q https://gist.githubusercontent.com/vkocaman/e091605f012ffc1efc0fcda170919602/raw/fae33d25bd026375b2aaf1194b68b9da559c4ac4/annotators.csv

dbutils.fs.cp("file:/databricks/driver/annotators.csv", "dbfs:/")

# COMMAND ----------

import pandas as pd

df = pd.read_csv("annotators.csv")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC By convention, there are three possible names:
# MAGIC 
# MAGIC Approach — Trainable annotator
# MAGIC 
# MAGIC Model — Trained annotator
# MAGIC 
# MAGIC nothing — Either a non-trainable annotator with pre-processing
# MAGIC step or shorthand for a model
# MAGIC 
# MAGIC So for example, Stemmer doesn’t say Approach nor Model, however, it is a Model. On the other hand, Tokenizer doesn’t say Approach nor Model, but it has a TokenizerModel(). Because it is not “training” anything, but it is doing some preprocessing before converting into a Model.
# MAGIC When in doubt, please refer to official documentation and API reference.
# MAGIC Even though we will do many hands-on practices in the following articles, let us give you a glimpse to let you understand the difference between AnnotatorApproach and AnnotatorModel.
# MAGIC As stated above, Tokenizer is an AnnotatorModel. So we need to call fit() and then transform().

# COMMAND ----------

# MAGIC %md
# MAGIC Now let’s see how this can be done in Spark NLP using Annotators and Transformers. Assume that we have the following steps that need to be applied one by one on a data frame.
# MAGIC 
# MAGIC - Split text into sentences
# MAGIC - Tokenize
# MAGIC - Normalize
# MAGIC - Get word embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC What’s actually happening under the hood?
# MAGIC 
# MAGIC When we fit() on the pipeline with Spark data frame (df), its text column is fed into DocumentAssembler() transformer at first and then a new column “document” is created in Document type (AnnotatorType). As we mentioned before, this transformer is basically the initial entry point to Spark NLP for any Spark data frame. Then its document column is fed into SentenceDetector() (AnnotatorApproach) and the text is split into an array of sentences and a new column “sentences” in Document type is created. Then “sentences” column is fed into Tokenizer() (AnnotatorModel) and each sentence is tokenized and a new column “token” in Token type is created. And so on.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Spark Dataframe

# COMMAND ----------

text = 'Peter Parker is a nice guy and lives in New York'

spark_df = spark.createDataFrame([[text]]).toDF("text")

spark_df.show(truncate=False)

# COMMAND ----------

from pyspark.sql.types import StringType, IntegerType

# if you want to create a spark datafarme from a list of strings

text_list = ['Peter Parker is a nice guy and lives in New York.', 'Bruce Wayne is also a nice guy and lives in Gotham City.']

spark.createDataFrame(text_list, StringType()).toDF("text").show(truncate=80)


# COMMAND ----------

from pyspark.sql import Row

spark.createDataFrame(list(map(lambda x: Row(text=x), text_list))).show(truncate=80)


# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/annotation/english/spark-nlp-basics/sample-sentences-en.txt
  
dbutils.fs.cp("file:/databricks/driver/sample-sentences-en.txt", "dbfs:/")

# COMMAND ----------

with open('sample-sentences-en.txt') as f:
  print (f.read())

# COMMAND ----------

spark_df = spark.read.text('/sample-sentences-en.txt').toDF('text')

spark_df.show(truncate=False)

# COMMAND ----------

spark_df.select('text').show(truncate=False)

# COMMAND ----------

# or we can even create a spark dataframe from pandas dataframe
temp_spark_df = spark.createDataFrame(df)

temp_spark_df.show()

# COMMAND ----------

temp_spark_df.createOrReplaceTempView("table1")

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC var scalaDF = spark.sql("select * from table1")
# MAGIC 
# MAGIC scalaDF.show(2)

# COMMAND ----------

pythonDF = spark.sql("select * from table1")

pythonDF.show(3)

# COMMAND ----------

textFiles = spark.sparkContext.wholeTextFiles("./*.txt",4)
    
spark_df_folder = textFiles.toDF(schema=['path','text'])

spark_df_folder.show(truncate=15)

# COMMAND ----------

spark_df_folder.select('text').take(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transformers

# COMMAND ----------

# MAGIC %md
# MAGIC What are we going to do if our DataFrame doesn’t have columns in those type? Here comes transformers. In Spark NLP, we have five different transformers that are mainly used for getting the data in or transform the data from one AnnotatorType to another. Here is the list of transformers:
# MAGIC 
# MAGIC `DocumentAssembler`: To get through the NLP process, we need to get raw data annotated. This is a special transformer that does this for us; it creates the first annotation of type Document which may be used by annotators down the road.
# MAGIC 
# MAGIC `TokenAssembler`: This transformer reconstructs a Document type annotation from tokens, usually after these have been, lemmatized, normalized, spell checked, etc, to use this document annotation in further annotators.
# MAGIC 
# MAGIC `Doc2Chunk`: Converts DOCUMENT type annotations into CHUNK type with the contents of a chunkCol.
# MAGIC 
# MAGIC `Chunk2Doc` : Converts a CHUNK type column back into DOCUMENT. Useful when trying to re-tokenize or do further analysis on a CHUNK result.
# MAGIC 
# MAGIC `Finisher`: Once we have our NLP pipeline ready to go, we might want to use our annotation results somewhere else where it is easy to use. The Finisher outputs annotation(s) values into a string.

# COMMAND ----------

# MAGIC %md
# MAGIC each annotator accepts certain types of columns and outputs new columns in another type (we call this AnnotatorType).
# MAGIC 
# MAGIC In Spark NLP, we have the following types: 
# MAGIC 
# MAGIC `Document`, `token`, `chunk`, `pos`, `word_embeddings`, `date`, `entity`, `sentiment`, `named_entity`, `dependency`, `labeled_dependency`. 
# MAGIC 
# MAGIC That is, the DataFrame you have needs to have a column from one of these types if that column will be fed into an annotator; otherwise, you’d need to use one of the Spark NLP transformers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Assembler

# COMMAND ----------

# MAGIC %md
# MAGIC In Spark NLP, we have five different transformers that are mainly used for getting the data in or transform the data from one AnnotatorType to another.

# COMMAND ----------

# MAGIC %md
# MAGIC That is, the DataFrame you have needs to have a column from one of these types if that column will be fed into an annotator; otherwise, you’d need to use one of the Spark NLP transformers. Here is the list of transformers: DocumentAssembler, TokenAssembler, Doc2Chunk, Chunk2Doc, and the Finisher.
# MAGIC 
# MAGIC So, let’s start with DocumentAssembler(), an entry point to Spark NLP annotators.

# COMMAND ----------

# MAGIC %md
# MAGIC To get through the process in Spark NLP, we need to get raw data transformed into Document type at first. 
# MAGIC 
# MAGIC DocumentAssembler() is a special transformer that does this for us; it creates the first annotation of type Document which may be used by annotators down the road.
# MAGIC 
# MAGIC DocumentAssembler() comes from sparknlp.base class and has the following settable parameters. See the full list here and the source code here.
# MAGIC 
# MAGIC `setInputCol()` -> the name of the column that will be converted. We can specify only one column here. It can read either a String column or an Array[String]
# MAGIC 
# MAGIC `setOutputCol()` -> optional : the name of the column in Document type that is generated. We can specify only one column here. Default is ‘document’
# MAGIC 
# MAGIC `setIdCol()` -> optional: String type column with id information
# MAGIC 
# MAGIC `setMetadataCol()` -> optional: Map type column with metadata information
# MAGIC 
# MAGIC `setCleanupMode()` -> optional: Cleaning up options, 
# MAGIC 
# MAGIC possible values:
# MAGIC ```
# MAGIC disabled: Source kept as original. This is a default.
# MAGIC inplace: removes new lines and tabs.
# MAGIC inplace_full: removes new lines and tabs but also those which were converted to strings (i.e. \n)
# MAGIC shrink: removes new lines and tabs, plus merging multiple spaces and blank lines to a single space.
# MAGIC shrink_full: remove new lines and tabs, including stringified values, plus shrinking spaces and blank lines.
# MAGIC ```

# COMMAND ----------

spark_df.show(truncate=False)

# COMMAND ----------

from sparknlp.base import *

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")\
  .setCleanupMode("shrink")

doc_df = documentAssembler.transform(spark_df)

doc_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC At first, we define DocumentAssembler with desired parameters and then transform the data frame with it. The most important point to pay attention to here is that you need to use a String or String[Array] type column in .setInputCol(). So it doesn’t have to be named as text. You just use the column name as it is.

# COMMAND ----------

doc_df.printSchema()

# COMMAND ----------

doc_df.select('document.result','document.begin','document.end').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC The new column is in an array of struct type and has the parameters shown above. The annotators and transformers all come with universal metadata that would be filled down the road depending on the annotators being used. Unless you want to append other Spark NLP annotators to DocumentAssembler(), you don’t need to know what all these parameters mean for now. So we will talk about them in the following articles. You can access all these parameters with {column name}.{parameter name}.
# MAGIC 
# MAGIC Let’s print out the first item’s result.

# COMMAND ----------

doc_df.select("document.result").take(1)

# COMMAND ----------

# MAGIC %md
# MAGIC If we would like to flatten the document column, we can do as follows.

# COMMAND ----------

import pyspark.sql.functions as F

doc_df.withColumn(
    "tmp", 
    F.explode("document"))\
    .select("tmp.*")\
    .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sentence Detector

# COMMAND ----------

# MAGIC %md
# MAGIC Finds sentence bounds in raw text.

# COMMAND ----------

# MAGIC %md
# MAGIC `setCustomBounds(string)`: Custom sentence separator text e.g. `["\n"]`
# MAGIC 
# MAGIC `setUseCustomOnly(bool)`: Use only custom bounds without considering those of Pragmatic Segmenter. Defaults to false. Needs customBounds.
# MAGIC 
# MAGIC `setUseAbbreviations(bool)`: Whether to consider abbreviation strategies for better accuracy but slower performance. Defaults to true.
# MAGIC 
# MAGIC `setExplodeSentences(bool)`: Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false.

# COMMAND ----------

from sparknlp.annotator import *

# we feed the document column coming from Document Assembler

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')


# COMMAND ----------

sent_df = sentenceDetector.transform(doc_df)

sent_df.show(truncate=50)

# COMMAND ----------

sent_df.select('sentences').take(3)

# COMMAND ----------

text ='The patient was prescribed 1 capsule of Advil for 5 days. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day. It was determined that all SGLT2 inhibitors should be discontinued indefinitely fro 3 months.'
text


# COMMAND ----------

spark_df = spark.createDataFrame([[text]]).toDF("text")

spark_df.show(truncate=False)

# COMMAND ----------

spark_df.show(truncate=100)

# COMMAND ----------

doc_df = documentAssembler.transform(spark_df)

sent_df = sentenceDetector.transform(doc_df)

sent_df.show(truncate=50)

# COMMAND ----------

sent_df.select('sentences.result').take(1)

# COMMAND ----------

sentenceDetector.setExplodeSentences(True)

# COMMAND ----------

sent_df = sentenceDetector.transform(doc_df)

sent_df.show(truncate=50)

# COMMAND ----------

sent_df.select('sentences.result').show(truncate=False)

# COMMAND ----------

from pyspark.sql import functions as F

sent_df.select(F.explode('sentences.result')).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentence Detector DL

# COMMAND ----------

sentencerDL = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

sent_dl_df = sentencerDL.transform(doc_df)

sent_dl_df.select(F.explode('sentences.result')).show(truncate=False)

# COMMAND ----------

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')
    
sentencerDL = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

sd_pipeline = PipelineModel(stages=[documenter, sentenceDetector])

sd_model = LightPipeline(sd_pipeline)

# DL version
sd_dl_pipeline = PipelineModel(stages=[documenter, sentencerDL])

sd_dl_model = LightPipeline(sd_dl_pipeline)


# COMMAND ----------

text = """John loves Mary.Mary loves Peter
Peter loves Helen .Helen loves John; 
Total: four people involved."""

for anno in sd_model.fullAnnotate(text)[0]["sentences"]:
    print("{}\t{}\t{}\t{}".format(
        anno.metadata["sentence"], anno.begin, anno.end, anno.result))

# COMMAND ----------

for anno in sd_dl_model.fullAnnotate(text)[0]["sentences"]:
    print("{}\t{}\t{}\t{}".format(
        anno.metadata["sentence"], anno.begin, anno.end, anno.result))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC Identifies tokens with tokenization open standards. It is an **Annotator Approach, so it requires .fit()**.
# MAGIC 
# MAGIC A few rules will help customizing it if defaults do not fit user needs.
# MAGIC 
# MAGIC setExceptions(StringArray): List of tokens to not alter at all. Allows composite tokens like two worded tokens that the user may not want to split.
# MAGIC 
# MAGIC `addException(String)`: Add a single exception
# MAGIC 
# MAGIC `setExceptionsPath(String)`: Path to txt file with list of token exceptions
# MAGIC 
# MAGIC `caseSensitiveExceptions(bool)`: Whether to follow case sensitiveness for matching exceptions in text
# MAGIC 
# MAGIC `contextChars(StringArray)`: List of 1 character string to rip off from tokens, such as parenthesis or question marks. Ignored if using prefix, infix or suffix patterns.
# MAGIC 
# MAGIC `splitChars(StringArray)`: List of 1 character string to split tokens inside, such as hyphens. Ignored if using infix, prefix or suffix patterns.
# MAGIC 
# MAGIC `splitPattern (String)`: pattern to separate from the inside of tokens. takes priority over splitChars.
# MAGIC setTargetPattern: Basic regex rule to identify a candidate for tokenization. Defaults to \\S+ which means anything not a space
# MAGIC 
# MAGIC `setSuffixPattern`: Regex to identify subtokens that are in the end of the token. Regex has to end with \\z and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
# MAGIC 
# MAGIC `setPrefixPattern`: Regex to identify subtokens that come in the beginning of the token. Regex has to start with \\A and must contain groups (). Each group will become a separate token within the prefix. Defaults to non-letter characters. e.g. quotes or parenthesis
# MAGIC 
# MAGIC `addInfixPattern`: Add an extension pattern regex with groups to the top of the rules (will target first, from more specific to the more general).
# MAGIC 
# MAGIC `minLength`: Set the minimum allowed legth for each token
# MAGIC 
# MAGIC `maxLength`: Set the maximum allowed legth for each token

# COMMAND ----------

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

# COMMAND ----------

text = 'Peter Parker (Spiderman) is a nice guy and lives in New York but has no e-mail!'

spark_df = spark.createDataFrame([[text]]).toDF("text")


# COMMAND ----------

doc_df = documentAssembler.transform(spark_df)

token_df = tokenizer.fit(doc_df).transform(doc_df)

token_df.show(truncate=50)

# COMMAND ----------

token_df.select('token.result').take(1)

# COMMAND ----------

tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")\
    .setSplitChars(['-'])\
    .setContextChars(['?', '!', '('])\
    .addException("New York")

# COMMAND ----------

token_df = tokenizer.fit(doc_df).transform(doc_df)

token_df.select('token.result').take(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regex Tokenizer

# COMMAND ----------

from pyspark.sql.types import StringType

content = "1. T1-T2 DATE**[12/24/13] $1.99 () (10/12), ph+ 90%"
pattern = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"

df = spark.createDataFrame([content], StringType()).withColumnRenamed("value", "text")

documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentence')

regexTokenizer = RegexTokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("regexToken") \
    .setPattern(pattern) \
    .setPositionalMask(False)

docPatternRemoverPipeline = Pipeline().setStages([
        documenter,
        sentenceDetector,
        regexTokenizer])

result = docPatternRemoverPipeline.fit(df).transform(df)

# COMMAND ----------

result.show(10,30)

# COMMAND ----------

import pyspark.sql.functions as F

result_df = result.select(F.explode('regexToken.result').alias('regexToken')).toPandas()
result_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stacking Spark NLP Annotators in Spark ML Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Spark NLP provides an easy API to integrate with Spark ML Pipelines and all the Spark NLP annotators and transformers can be used within Spark ML Pipelines. So, it’s better to explain Pipeline concept through Spark ML official documentation.
# MAGIC 
# MAGIC What is a Pipeline anyway? In machine learning, it is common to run a sequence of algorithms to process and learn from data. 
# MAGIC 
# MAGIC Apache Spark ML represents such a workflow as a Pipeline, which consists of a sequence of PipelineStages (Transformers and Estimators) to be run in a specific order.
# MAGIC 
# MAGIC In simple terms, a pipeline chains multiple Transformers and Estimators together to specify an ML workflow. We use Pipeline to chain multiple Transformers and Estimators together to specify our machine learning workflow.
# MAGIC 
# MAGIC The figure below is for the training time usage of a Pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC A Pipeline is specified as a sequence of stages, and each stage is either a Transformer or an Estimator. These stages are run in order, and the input DataFrame is transformed as it passes through each stage. That is, the data are passed through the fitted pipeline in order. Each stage’s transform() method updates the dataset and passes it to the next stage. With the help of Pipelines, we can ensure that training and test data go through identical feature processing steps.
# MAGIC 
# MAGIC Now let’s see how this can be done in Spark NLP using Annotators and Transformers. Assume that we have the following steps that need to be applied one by one on a data frame.
# MAGIC 
# MAGIC - Split text into sentences
# MAGIC - Tokenize
# MAGIC 
# MAGIC And here is how we code this pipeline up in Spark NLP.

# COMMAND ----------

from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     sentenceDetector,
     tokenizer
 ])

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

# COMMAND ----------

spark_df = spark.read.text('/sample-sentences-en.txt').toDF('text')

spark_df.show(truncate=False)

# COMMAND ----------

result = pipelineModel.transform(spark_df)

# COMMAND ----------

result.show(truncate=20)

# COMMAND ----------

result.printSchema()

# COMMAND ----------

result.select('sentences.result').take(3)

# COMMAND ----------

result.select('token').take(3)[2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalizer

# COMMAND ----------

# MAGIC %md
# MAGIC Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary
# MAGIC 
# MAGIC `setCleanupPatterns(patterns)`: Regular expressions list for normalization, defaults [^A-Za-z]
# MAGIC 
# MAGIC `setLowercase(value)`: lowercase tokens, default false
# MAGIC 
# MAGIC `setSlangDictionary(path)`: txt file with delimited words to be transformed into something else

# COMMAND ----------

import string
string.punctuation

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
    
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")\
    .setLowercase(True)\
    .setCleanupPatterns(["[^\w\d\s]"]) # remove punctuations (keep alphanumeric chars)
    # if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])


nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     normalizer])

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

# COMMAND ----------

pipelineModel.stages

# COMMAND ----------

result = pipelineModel.transform(spark_df)

# COMMAND ----------

result.show(truncate=20)

# COMMAND ----------

result.select('token').take(3)

# COMMAND ----------

result.select('normalized.result').take(2)

# COMMAND ----------

result.select('normalized').take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Document Normalizer

# COMMAND ----------

# MAGIC %md
# MAGIC The DocumentNormalizer is an annotator that can be used after the DocumentAssembler to narmalize documents once that they have been processed and indexed .
# MAGIC It takes in input annotated documents of type Array AnnotatorType.DOCUMENT and gives as output annotated document of type AnnotatorType.DOCUMENT .
# MAGIC 
# MAGIC Parameters are:  
# MAGIC - inputCol: input column name string which targets a column of type Array(AnnotatorType.DOCUMENT).   
# MAGIC - outputCol: output column name string which targets a column of type AnnotatorType.DOCUMENT.  
# MAGIC - action: action string to perform applying regex patterns, i.e. (clean | extract). Default is "clean".  
# MAGIC - cleanupPatterns: normalization regex patterns which match will be removed from document. Default is "<[^>]*>" (e.g., it removes all HTML tags).  
# MAGIC - replacement: replacement string to apply when regexes match. Default is " ".  
# MAGIC - lowercase: whether to convert strings to lowercase. Default is False.  
# MAGIC - removalPolicy: removalPolicy to remove patterns from text with a given policy. Valid policy values are: "all", "pretty_all", "first", "pretty_first". Defaults is "pretty_all".  
# MAGIC - encoding: file encoding to apply on normalized documents. Supported encodings are: UTF_8, UTF_16, US_ASCII, ISO-8859-1, UTF-16BE, UTF-16LE. Default is "UTF-8".

# COMMAND ----------

text = '''
  <div id="theworldsgreatest" class='my-right my-hide-small my-wide toptext' style="font-family:'Segoe UI',Arial,sans-serif">
    THE WORLD'S LARGEST WEB DEVELOPER SITE
    <h1 style="font-size:300%;">THE WORLD'S LARGEST WEB DEVELOPER SITE</h1>
    <p style="font-size:160%;">Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum..</p>
  </div>

</div>'''

# COMMAND ----------

spark_df = spark.createDataFrame([[text]]).toDF("text")

spark_df.show(truncate=False)

# COMMAND ----------

documentNormalizer = DocumentNormalizer() \
    .setInputCols("document") \
    .setOutputCol("normalizedDocument")

documentNormalizer.extractParamMap()

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

#default
cleanUpPatterns = ["<[^>]*>"]

documentNormalizer = DocumentNormalizer() \
    .setInputCols("document") \
    .setOutputCol("normalizedDocument") \
    .setAction("clean") \
    .setPatterns(cleanUpPatterns) \
    .setReplacement(" ") \
    .setPolicy("pretty_all") \
    .setLowercase(True)

docPatternRemoverPipeline = Pipeline() \
    .setStages([
        documentAssembler,
        documentNormalizer])
    

pipelineModel = docPatternRemoverPipeline.fit(spark_df).transform(spark_df)

# COMMAND ----------

pipelineModel.select('normalizedDocument.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC for more examples : https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/document-normalizer/document_normalizer_notebook.ipynb

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stopwords Cleaner

# COMMAND ----------

# MAGIC %md
# MAGIC This annotator excludes from a sequence of strings (e.g. the output of a Tokenizer, Normalizer, Lemmatizer, and Stemmer) and drops all the stop words from the input sequences.

# COMMAND ----------

# MAGIC %md
# MAGIC Functions:
# MAGIC 
# MAGIC `setStopWords`: The words to be filtered out. Array[String]
# MAGIC 
# MAGIC `setCaseSensitive`: Whether to do a case sensitive comparison over the stop words.

# COMMAND ----------

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("token")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)\
      #.setStopWords(["no", "without"]) (e.g. read a list of words from a txt)

# COMMAND ----------

stopwords_cleaner.getStopWords()

# COMMAND ----------

documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

tokenizer = Tokenizer() \
      .setInputCols(["document"]) \
      .setOutputCol("token")

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("token")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)\

nlpPipeline = Pipeline(stages=[
 documentAssembler, 
 tokenizer,
 stopwords_cleaner
 ])

spark_df = spark.read.text('/sample-sentences-en.txt').toDF('text')

result = nlpPipeline.fit(spark_df).transform(spark_df)

result.show(40)

# COMMAND ----------

result.select('cleanTokens.result').take(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Token Assembler

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")\
    .setLowercase(False)\

stopwords_cleaner = StopWordsCleaner()\
    .setInputCols("normalized")\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)\

tokenassembler = TokenAssembler()\
    .setInputCols(["sentences", "cleanTokens"]) \
    .setOutputCol("clean_text")


nlpPipeline = Pipeline(stages=[
     documentAssembler,
     sentenceDetector,
     tokenizer,
     normalizer,
     stopwords_cleaner,
     tokenassembler
 ])

result = nlpPipeline.fit(spark_df).transform(spark_df)

result.show()

# COMMAND ----------

result.select('clean_text').take(1)

# COMMAND ----------

# if we use TokenAssembler().setPreservePosition(True), the original borders will be preserved (dropped & unwanted chars will be replaced by spaces)

result.select('clean_text').take(1)

# COMMAND ----------

result.select('text', F.explode('clean_text.result').alias('clean_text')).show(truncate=False)

# COMMAND ----------

import pyspark.sql.functions as F

result.withColumn(
    "tmp", 
    F.explode("clean_text")) \
    .select("tmp.*").select("begin","end","result","metadata.sentence").show(truncate = False)

# COMMAND ----------

result.select('text', F.explode('clean_text.result').alias('clean_text')).toPandas()

# COMMAND ----------

# if we hadn't used Sentence Detector, this would be what we got. (tokenizer gets document instead of sentences column)

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

tokenassembler = TokenAssembler()\
    .setInputCols(["document", "cleanTokens"]) \
    .setOutputCol("clean_text")

nlpPipeline = Pipeline(stages=[
     documentAssembler,
     tokenizer,
     normalizer,
     stopwords_cleaner,
     tokenassembler
 ])

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

result = pipelineModel.transform(spark_df)

result.select('text', 'clean_text.result').show(truncate=False)

# COMMAND ----------

result.withColumn(
    "tmp", 
    F.explode("clean_text")) \
    .select("tmp.*").select("begin","end","result","metadata.sentence").show(truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC **IMPORTANT NOTE:**
# MAGIC 
# MAGIC If you have some other steps & annotators in your pipeline that will need to use the tokens from cleaned text (assembled tokens), you will need to tokenize the processed text again as the original text is probably changed completely.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stemmer

# COMMAND ----------

# MAGIC %md
# MAGIC Returns hard-stems out of words with the objective of retrieving the meaningful part of the word

# COMMAND ----------

stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     stemmer
 ])

result = nlpPipeline.fit(spark_df).transform(spark_df)

result.show()

# COMMAND ----------

result.select('stem.result').show(truncate=False)

# COMMAND ----------

import pyspark.sql.functions as F

result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.stem.result)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("stem")).toPandas()

result_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lemmatizer

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieves lemmas out of words with the objective of returning a base dictionary word

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/mahavivo/vocabulary/master/lemmas/AntBNC_lemmas_ver_001.txt
  
dbutils.fs.cp("file:/databricks/driver/AntBNC_lemmas_ver_001.txt", "dbfs:/")

# COMMAND ----------

lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("dbfs:/AntBNC_lemmas_ver_001.txt", value_delimiter ="\t", key_delimiter = "->")

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     stemmer,
     lemmatizer
 ])

result = nlpPipeline.fit(spark_df).transform(spark_df)
result.show()

# COMMAND ----------

result.select('lemma.result').show(truncate=False)

# COMMAND ----------

result_df = result.select(F.explode(F.arrays_zip(result.token.result, result.stem.result, result.lemma.result)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("stem"),
                          F.expr("cols['2']").alias("lemma")).toPandas()

result_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## NGram Generator

# COMMAND ----------

# MAGIC %md
# MAGIC NGramGenerator annotator takes as input a sequence of strings (e.g. the output of a `Tokenizer`, `Normalizer`, `Stemmer`, `Lemmatizer`, and `StopWordsCleaner`). 
# MAGIC 
# MAGIC The parameter n is used to determine the number of terms in each n-gram. The output will consist of a sequence of n-grams where each n-gram is represented by a space-delimited string of n consecutive words with annotatorType `CHUNK` same as the Chunker annotator.
# MAGIC 
# MAGIC Functions:
# MAGIC 
# MAGIC `setN:` number elements per n-gram (>=1)
# MAGIC 
# MAGIC `setEnableCumulative:` whether to calculate just the actual n-grams or all n-grams from 1 through n
# MAGIC 
# MAGIC `setDelimiter:` Glue character used to join the tokens

# COMMAND ----------

ngrams_cum = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams") \
            .setN(3) \
            .setEnableCumulative(True)\
            .setDelimiter("_") # Default is space
    
# .setN(3) means, take bigrams and trigrams.

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     ngrams_cum])

result = nlpPipeline.fit(spark_df).transform(spark_df)
result.select('ngrams.result').show(truncate=150)

# COMMAND ----------

ngrams_nonCum = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams_v2") \
            .setN(3) \
            .setEnableCumulative(False)\
            .setDelimiter("_") # Default is space
    
ngrams_nonCum.transform(result).select('ngrams_v2.result').show(truncate=150)

# COMMAND ----------

# MAGIC %md
# MAGIC ## TextMatcher

# COMMAND ----------

# MAGIC %md
# MAGIC Annotator to match entire phrases (by token) provided in a file against a Document
# MAGIC 
# MAGIC Functions:
# MAGIC 
# MAGIC `setEntities(path, format, options)`: Provides a file with phrases to match. Default: Looks up path in configuration.
# MAGIC 
# MAGIC `path`: a path to a file that contains the entities in the specified format.
# MAGIC 
# MAGIC `readAs`: the format of the file, can be one of {ReadAs.LINE_BY_LINE, ReadAs.SPARK_DATASET}. Defaults to LINE_BY_LINE.
# MAGIC 
# MAGIC `options`: a map of additional parameters. Defaults to {“format”: “text”}.
# MAGIC 
# MAGIC `entityValue` : Value for the entity metadata field to indicate which chunk comes from which textMatcher when there are multiple textMatchers. 
# MAGIC 
# MAGIC `mergeOverlapping` : whether to merge overlapping matched chunks. Defaults false
# MAGIC 
# MAGIC `caseSensitive` : whether to match regardless of case. Defaults true

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/news_category_train.csv

dbutils.fs.cp("file:/databricks/driver/news_category_train.csv", "dbfs:/")

# COMMAND ----------

news_df = spark.read \
      .option("header", True) \
      .csv("/news_category_train.csv")

news_df.show(5, truncate=50)

# COMMAND ----------

# write the target entities to txt file 

entities = ['Wall Street', 'USD', 'stock', 'NYSE']
with open ('financial_entities.txt', 'w') as f:
    for i in entities:
        f.write(i+'\n')


entities = ['soccer', 'world cup', 'Messi', 'FC Barcelona']
with open ('sport_entities.txt', 'w') as f:
    for i in entities:
        f.write(i+'\n')


# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/financial_entities.txt", "dbfs:/")
dbutils.fs.cp("file:/databricks/driver/sport_entities.txt", "dbfs:/")

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("description")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

financial_entity_extractor = TextMatcher() \
    .setInputCols(["document",'token'])\
    .setOutputCol("financial_entities")\
    .setEntities("file:/databricks/driver/financial_entities.txt")\
    .setCaseSensitive(False)\
    .setEntityValue('financial_entity')

sport_entity_extractor = TextMatcher() \
    .setInputCols(["document",'token'])\
    .setOutputCol("sport_entities")\
    .setEntities("file:/databricks/driver/sport_entities.txt")\
    .setCaseSensitive(False)\
    .setEntityValue('sport_entity')


nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     financial_entity_extractor,
     sport_entity_extractor
 ])

result = nlpPipeline.fit(news_df).transform(news_df)

# COMMAND ----------

result.select('financial_entities.result','sport_entities.result').take(2)

# COMMAND ----------

result.select('description','financial_entities.result','sport_entities.result')\
      .toDF('text','financial_matches','sport_matches').filter((F.size('financial_matches')>1) | (F.size('sport_matches')>1))\
      .show(truncate=70)


# COMMAND ----------

result_df = result.select(F.explode(F.arrays_zip(result.financial_entities.result, 
                                                 result.financial_entities.begin, 
                                                 result.financial_entities.end)).alias("cols")) \
                  .select(F.expr("cols['0']").alias("clinical_entities"),
                          F.expr("cols['1']").alias("begin"),
                          F.expr("cols['2']").alias("end")).toPandas()

result_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## RegexMatcher

# COMMAND ----------

! wget -q https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/pubmed/pubmed-sample.csv

dbutils.fs.cp("file:/databricks/driver/pubmed-sample.csv", "dbfs:/")

# COMMAND ----------

pubMedDF = spark.read\
                .option("header", "true")\
                .csv("/pubmed-sample.csv")\
                .filter("AB IS NOT null")\
                .withColumnRenamed("AB", "text")\
                .drop("TI")

pubMedDF.show(truncate=50)

# COMMAND ----------

rules = '''
renal\s\w+, started with 'renal'
cardiac\s\w+, started with 'cardiac'
\w*ly\b, ending with 'ly'
\S*\d+\S*, match any word that contains numbers
(\d+).?(\d*)\s*(mg|ml|g), match medication metrics
'''

with open('regex_rules.txt', 'w') as f:
    
    f.write(rules)

dbutils.fs.cp("file:/databricks/driver/regex_rules.txt", "dbfs:/")

# COMMAND ----------

RegexMatcher().extractParamMap()

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

regex_matcher = RegexMatcher()\
    .setInputCols('document')\
    .setStrategy("MATCH_ALL")\
    .setOutputCol("regex_matches")\
    .setExternalRules(path="file:/databricks/driver/regex_rules.txt", delimiter=',')
    
nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     regex_matcher
 ])


match_df = nlpPipeline.fit(pubMedDF).transform(pubMedDF)
match_df.select('regex_matches.result').take(3)

# COMMAND ----------

match_df.select('text','regex_matches.result')\
        .toDF('text','matches').filter(F.size('matches')>1)\
        .show(truncate=70)


# COMMAND ----------

# MAGIC %md
# MAGIC ## MultiDateMatcher

# COMMAND ----------

# MAGIC %md
# MAGIC Extract exact & normalize dates from relative date-time phrases. The default anchor date will be the date the code is run.

# COMMAND ----------

MultiDateMatcher().extractParamMap()

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

date_matcher = MultiDateMatcher() \
    .setInputCols('document') \
    .setOutputCol("date") \
    .setOutputFormat("yyyy/MM/dd")\
    .setSourceLanguage("en")
        
date_pipeline = PipelineModel(stages=[
     documentAssembler, 
     date_matcher
 ])

sample_df = spark.createDataFrame([['I saw him yesterday and he told me that he will visit us next week']]).toDF("text")

result = date_pipeline.transform(sample_df)

result.select('date.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's set the Input Format and Output Format to specific formatLet's set the Input Format and Output Format to specific format

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

date_matcher = MultiDateMatcher() \
    .setInputCols('document') \
    .setOutputCol("date")\
    .setInputFormats(["dd/MM/yyyy"])\
    .setOutputFormat("yyyy/MM/dd")\
    .setSourceLanguage("en")

date_pipeline = PipelineModel(stages=[
     documentAssembler, 
     date_matcher
 ])

sample_df = spark.createDataFrame([["the last payment date of this invoice is 21/05/2022"]]).toDF("text")

result = date_pipeline.transform(sample_df)

result.select('date.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Cleaning with UDF

# COMMAND ----------

text = '<h1 style="color: #5e9ca0;">Have a great <span  style="color: #2b2301;">birth</span> day!</h1>'

text_df = spark.createDataFrame([[text]]).toDF("text")

import re
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

clean_text = lambda s: re.sub(r'<[^>]*>', '', s)

text_df.withColumn('cleaned', udf(clean_text, StringType())('text')).select('text','cleaned').show(truncate= False)

# COMMAND ----------

find_not_alnum_count = lambda s: len([i for i in s if not i.isalnum() and i!=' '])

find_not_alnum_count("it's your birth day!")

# COMMAND ----------

text = '<h1 style="color: #5e9ca0;">Have a great <span  style="color: #2b2301;">birth</span> day!</h1>'

find_not_alnum_count(text)

# COMMAND ----------

text_df.withColumn('cleaned', udf(find_not_alnum_count, IntegerType())('text')).select('text','cleaned').show(truncate= False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Finisher

# COMMAND ----------

# MAGIC %md
# MAGIC ***Finisher:*** Once we have our NLP pipeline ready to go, we might want to use our annotation results somewhere else where it is easy to use. The Finisher outputs annotation(s) values into a string.
# MAGIC 
# MAGIC If we just want the desired output column in the final dataframe, we can use Finisher to drop previous stages in the final output and get the `result` from the process.
# MAGIC 
# MAGIC This is very handy when you want to use the output from Spark NLP annotator as an input to another Spark ML transformer.
# MAGIC 
# MAGIC Settable parameters are:
# MAGIC 
# MAGIC `setInputCols()`
# MAGIC 
# MAGIC `setOutputCols()`
# MAGIC 
# MAGIC `setCleanAnnotations(True)` -> Whether to remove intermediate annotations
# MAGIC 
# MAGIC `setValueSplitSymbol(“#”)` -> split values within an annotation character
# MAGIC 
# MAGIC `setAnnotationSplitSymbol(“@”)` -> split values between annotations character
# MAGIC 
# MAGIC `setIncludeMetadata(False)` -> Whether to include metadata keys. Sometimes useful in some annotations.
# MAGIC 
# MAGIC `setOutputAsArray(False)` -> Whether to output as Array. Useful as input for other Spark transformers.

# COMMAND ----------

finisher = Finisher() \
    .setInputCols(["regex_matches"]) \
    .setIncludeMetadata(False) # set to False to remove metadata

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     regex_matcher,
     finisher
 ])

match_df = nlpPipeline.fit(pubMedDF).transform(pubMedDF)
match_df.show(truncate = 50)

# COMMAND ----------

match_df.printSchema()

# COMMAND ----------

match_df.filter(F.size('finished_regex_matches')>2).show(truncate = 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LightPipeline
# MAGIC 
# MAGIC https://medium.com/spark-nlp/spark-nlp-101-lightpipeline-a544e93f20f1

# COMMAND ----------

# MAGIC %md
# MAGIC LightPipelines are Spark NLP specific Pipelines, equivalent to Spark ML Pipeline, but meant to deal with smaller amounts of data. They’re useful working with small datasets, debugging results, or when running either training or prediction from an API that serves one-off requests.
# MAGIC 
# MAGIC Spark NLP LightPipelines are Spark ML pipelines converted into a single machine but the multi-threaded task, becoming more than 10x times faster for smaller amounts of data (small is relative, but 50k sentences are roughly a good maximum). To use them, we simply plug in a trained (fitted) pipeline and then annotate a plain text. We don't even need to convert the input text to DataFrame in order to feed it into a pipeline that's accepting DataFrame as an input in the first place. This feature would be quite useful when it comes to getting a prediction for a few lines of text from a trained ML model.
# MAGIC 
# MAGIC  **It is nearly 10x faster than using Spark ML Pipeline**
# MAGIC 
# MAGIC `LightPipeline(someTrainedPipeline).annotate(someStringOrArray)`

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")

lemmatizer = Lemmatizer() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma") \
    .setDictionary("dbfs:/AntBNC_lemmas_ver_001.txt", value_delimiter ="\t", key_delimiter = "->")

nlpPipeline = Pipeline(stages=[
     documentAssembler, 
     tokenizer,
     stemmer,
     lemmatizer
 ])

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

pipelineModel.transform(spark_df).show()

# COMMAND ----------

from sparknlp.base import LightPipeline

light_model = LightPipeline(pipelineModel)

light_result = light_model.annotate("John and Peter are brothers. However they don't support each other that much.")

# COMMAND ----------

light_result.keys()

# COMMAND ----------

list(zip(light_result['token'], light_result['stem'], light_result['lemma']))

# COMMAND ----------

light_result = light_model.fullAnnotate("John and Peter are brothers. However they don't support each other that much.")

# COMMAND ----------

light_result

# COMMAND ----------

text_list= ["How did serfdom develop in and then leave Russia ?",
"There will be some exciting breakthroughs in NLP this year."]

light_model.annotate(text_list)

# COMMAND ----------

# MAGIC %md
# MAGIC **important note:** When you use Finisher in your pipeline, regardless of setting `cleanAnnotations` to False or True, LightPipeline will only return the finished columns.

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #