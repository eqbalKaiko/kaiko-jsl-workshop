# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Unsupervised Keyword Extraction

# COMMAND ----------

# MAGIC %md
# MAGIC # Keyword Extraction with YAKE

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.types import StringType, DataType,ArrayType
from pyspark.sql.functions import udf, struct
from IPython.core.display import display, HTML
import re

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

print("Spark NLP version", sparknlp.version())

print("Apache Spark version:", spark.version)

spark

# COMMAND ----------

stopwords = StopWordsCleaner().getStopWords()

# COMMAND ----------

stopwords[:5]

# COMMAND ----------

# MAGIC %md
# MAGIC ## YAKE Keyword Extractor
# MAGIC 
# MAGIC Yake is an Unsupervised, Corpus-Independent, Domain and Language-Independent and Single-Document keyword extraction algorithm.
# MAGIC 
# MAGIC Extracting keywords from texts has become a challenge for individuals and organizations as the information grows in complexity and size. The need to automate this task so that text can be processed in a timely and adequate manner has led to the emergence of automatic keyword extraction tools. Yake is a novel feature-based system for multi-lingual keyword extraction, which supports texts of different sizes, domain or languages. Unlike other approaches, Yake does not rely on dictionaries nor thesauri, neither is trained against any corpora. Instead, it follows an unsupervised approach which builds upon features extracted from the text, making it thus applicable to documents written in different languages without the need for further knowledge. This can be beneficial for a large number of tasks and a plethora of situations where access to training corpora is either limited or restricted.
# MAGIC 
# MAGIC 
# MAGIC The algorithm makes use of the position of a sentence and token. Therefore, to use the annotator, the text should be first sent through a Sentence Boundary Detector and then a tokenizer.
# MAGIC 
# MAGIC You can tweak the following parameters to get the best result from the annotator.
# MAGIC 
# MAGIC - *setMinNGrams(int)* Select the minimum length of a extracted keyword
# MAGIC - *setMaxNGrams(int)* Select the maximum length of a extracted keyword
# MAGIC - *setNKeywords(int)* Extract the top N keywords
# MAGIC - *setStopWords(list)* Set the list of stop words
# MAGIC - *setThreshold(float)* Each keyword will be given a keyword score greater than 0. (Lower the score better the keyword) Set an upper bound for the keyword score from this method.
# MAGIC - *setWindowSize(int)* Yake will construct a co-occurence matrix. You can set the window size for the cooccurence matrix construction from this method. ex: windowSize=2 will look at two words to both left and right of a candidate word.
# MAGIC 
# MAGIC 
# MAGIC <b>References</b>
# MAGIC 
# MAGIC Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp 257-289. [pdf](https://doi.org/10.1016/j.ins.2019.09.013)

# COMMAND ----------

document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols("document") \
    .setOutputCol("sentence")

token = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token") \
    .setContextChars(["(", ")", "?", "!", ".", ","])

keywords = YakeKeywordExtraction() \
    .setInputCols("token") \
    .setOutputCol("keywords") \
    .setMinNGrams(1) \
    .setMaxNGrams(3)\
    .setNKeywords(20)\
    .setStopWords(stopwords)

yake_pipeline = Pipeline(stages=[document, 
                                 sentenceDetector, 
                                 token, keywords])

empty_df = spark.createDataFrame([['']]).toDF("text")

yake_Model = yake_pipeline.fit(empty_df)


# COMMAND ----------

# LightPipeline

light_model = LightPipeline(yake_Model)

text = '''
google is acquiring data science community kaggle. Sources tell us that google is acquiring kaggle, a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat vague , but given that google is hosting its Cloud Next conference in san francisco this week, the official announcement could come as early as tomorrow. Reached by phone, kaggle co-founder ceo anthony goldbloom declined to deny that the acquisition is happening. google itself declined 'to comment on rumors'. kaggle, which has about half a million data scientists on its platform, was founded by Goldbloom and Ben Hamner in 2010. The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well ahead of them by focusing on its specific niche. The service is basically the de facto home for running data science and machine learning competitions. With kaggle, google is buying one of the largest and most active communities for data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). kaggle has a bit of a history with google, too, but that's pretty recent. Earlier this month, google and kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That competition had some deep integrations with the google Cloud platform, too. Our understanding is that google will keep the service running - likely under its current name. While the acquisition is probably more about Kaggle's community than technology, kaggle did build some interesting tools for hosting its competition and 'kernels', too. On kaggle, kernels are basically the source code for analyzing data sets and developers can share this code on the platform (the company previously called them 'scripts'). Like similar competition-centric sites, kaggle also runs a job board, too. It's unclear what google will do with that part of the service. According to Crunchbase, kaggle raised $12.5 million (though PitchBook says it's $12.75) since its launch in 2010. Investors in kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, google chief economist Hal Varian, Khosla Ventures and Yuri Milner
'''

light_result = light_model.fullAnnotate(text)[0]

[(s.metadata['sentence'], s.result) for s in light_result['sentence']]

# COMMAND ----------

import pandas as pd

keys_df = pd.DataFrame([(k.result, k.begin, k.end, k.metadata['score'],  k.metadata['sentence']) for k in light_result['keywords']],
                       columns = ['keywords','begin','end','score','sentence'])
keys_df['score'] = keys_df['score'].astype(float)

# ordered by relevance 
keys_df.sort_values(['sentence','score']).head(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting keywords from datraframe

# COMMAND ----------

! wget -q https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/pubmed/pubmed_sample_text_small.csv
  
dbutils.fs.cp("file:/databricks/driver/pubmed_sample_text_small.csv", "dbfs:/")

# COMMAND ----------

df = spark.read\
                .option("header", "true")\
                .csv("/pubmed_sample_text_small.csv")\
                
df.show(truncate=50)

# COMMAND ----------

result = yake_pipeline.fit(df).transform(df)

# COMMAND ----------

result = result.withColumn('unique_keywords', F.array_distinct("keywords.result"))

# COMMAND ----------

result.show(1)

# COMMAND ----------

result.select('keywords.result').show(1,truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #