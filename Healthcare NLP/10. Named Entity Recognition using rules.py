# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Named Entity Recognition using rules

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

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)  
pd.set_option('display.expand_frame_repr', False)


print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC # How the ContextualParser Works

# COMMAND ----------

# MAGIC %md
# MAGIC Spark NLP's `ContextualParser` is a licensed annotator that allows users to extract entities from a document based on pattern matching. It provides more functionality than its open-source counterpart `EntityRuler` by allowing users to customize specific characteristics for pattern matching. You're able to find entities using regex rules for full and partial matches, a dictionary with normalizing options and context parameters to take into account things such as token distances. 
# MAGIC 
# MAGIC There are 3 components necessary to understand when using the `ContextualParser` annotator:
# MAGIC 
# MAGIC 1. `ContextualParser` annotator's parameters
# MAGIC 2. JSON configuration file
# MAGIC 3. Dictionary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ContextualParser Annotator Parameters

# COMMAND ----------

# MAGIC %md
# MAGIC Here are all the parameters available to use with the `ContextualParserApproach`:
# MAGIC   
# MAGIC ```
# MAGIC contextualParser = ContextualParserApproach() \
# MAGIC     .setInputCols(["sentence", "token"]) \
# MAGIC     .setOutputCol("entity") \
# MAGIC     .setCaseSensitive(True) \
# MAGIC     .setJsonPath("context_config.json") \
# MAGIC     .setPrefixAndSuffixMatch(True) \
# MAGIC     .setCompleteContextMatch(True) \
# MAGIC     .setDictionary("dictionary.tsv", options={"orientation":"vertical"})
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC We will dive deeper into the details of each parameter, but here's a quick overview:
# MAGIC 
# MAGIC - `setCaseSensitive`: do you want the matching to be case sensitive (applies to all JSON properties apart from the regex property)
# MAGIC - `setJsonPath`: the path to your JSON configuration file
# MAGIC - `setPrefixAndSuffixMatch`: do you want to match using both the prefix AND suffix properties from the JSON configuration file
# MAGIC - `setCompleteContextMatch`: do you want an exact match of prefix and suffix.
# MAGIC - `setDictionary`: the path to your dictionary, used for normalizing entities
# MAGIC 
# MAGIC Let's start by looking at the JSON configuration file.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. JSON Configuration File

# COMMAND ----------

# MAGIC %md
# MAGIC Here is a fully utilized JSON configuration file.
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Gender",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "girl|boy",
# MAGIC   "completeMatchRegex": "true",
# MAGIC   "matchScope": "token",
# MAGIC   "prefix": ["birth", "growing", "assessment"],
# MAGIC   "suffix": ["faster", "velocities"],
# MAGIC   "contextLength": 100,
# MAGIC   "contextException": ["slightly"],
# MAGIC   "exceptionDistance": 40
# MAGIC  }
# MAGIC  ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Basic Properties

# COMMAND ----------

# MAGIC %md
# MAGIC There are 5 basic properties you can set in your JSON configuration file:
# MAGIC 
# MAGIC - `entity`
# MAGIC - `ruleScope`
# MAGIC - `regex`
# MAGIC - `completeMatchRegex`
# MAGIC - `matchScope`
# MAGIC 
# MAGIC Let's first look at the 3 most essential properties to set:
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Digit",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "\\d+" # Note here: backslashes are escape characters in JSON, so for regex pattern "\d+" we need to write it out as "\\d+"
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we're looking for tokens in our text that match the regex: "`\d+`" and assign the "`Digit`" entity to those tokens. When `ruleScope` is set to "`sentence`", we're looking for a match on each *token* of a **sentence**. You can change it to "`document`" to look for a match on each *sentence* of a **document**. The latter is particularly useful when working with multi-word matches, but we'll explore this at a later stage.
# MAGIC 
# MAGIC The next properties to look at are `completeMatchRegex` and `matchScope`. To understand their use case, let's take a look at an example where we're trying to match all digits in our text. 
# MAGIC 
# MAGIC Let's say we come across the following string: ***XYZ987***
# MAGIC 
# MAGIC Depending on how we set the `completeMatchRegex` and `matchScope` properties, we'll get the following results:
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Digit",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "\\d+",
# MAGIC   "completeMatchRegex": "false",
# MAGIC   "matchScope": "token"
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC `OUTPUT: [XYZ987]`
# MAGIC 
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Digit",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "\\d+",  
# MAGIC   "completeMatchRegex": "false",
# MAGIC   "matchScope": "sub-token"
# MAGIC }
# MAGIC ```
# MAGIC `OUTPUT: [987]`
# MAGIC 
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Digit",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "\\d+",
# MAGIC   "completeMatchRegex": "true"
# MAGIC   # matchScope is ignored here
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC `OUTPUT: []`

# COMMAND ----------

# MAGIC %md
# MAGIC `"completeMatchRegex": "true"` will only return an output if our string was modified in the following way (to get a complete, exact match): **XYZ 987**
# MAGIC 
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Digit",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "\\d+",  
# MAGIC   "completeMatchRegex": "true",
# MAGIC   "matchScope": "token" # Note here: sub-token would return the same output
# MAGIC }
# MAGIC ```
# MAGIC `OUTPUT: [987]`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2. Context Awareness Properties
# MAGIC 
# MAGIC There are 5 properties related to context awareness:
# MAGIC 
# MAGIC - `contextLength`
# MAGIC - `prefix`
# MAGIC - `suffix`
# MAGIC - `contextException`
# MAGIC - `exceptionDistance`

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at a similar example. Say we have the following text: ***At birth, the typical boy is growing slightly faster than the typical girl, but growth rates become equal at about seven months.***
# MAGIC 
# MAGIC If we want to match the gender that grows faster at birth, we can start by defining our regex: "`girl|boy`"
# MAGIC 
# MAGIC Next, we add a prefix ("`birth`") and suffix ("`faster`") to ask the parser to match the regex only if the word "`birth`" comes before and only if the word "`faster`" comes after. Finally, we will need to set the `contextLength` - this is the maximum number of tokens after the prefix and before the suffix that will be searched to find a regex match.
# MAGIC 
# MAGIC Here's what the JSON configuration file would look like:
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Gender",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "girl|boy",
# MAGIC   "contextLength": 50,
# MAGIC   "prefix": ["birth"],
# MAGIC   "suffix": ["faster"]
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC `OUTPUT: [boy]`

# COMMAND ----------

# MAGIC %md
# MAGIC If you remember, the annotator has a `setPrefixAndSuffixMatch()` parameter. If you set it to `True`, the previous output would remain as is. However, if you had set it to `False` and used the following JSON configuration:
# MAGIC   
# MAGIC   ```
# MAGIC {
# MAGIC   "entity": "Gender",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "girl|boy",
# MAGIC   "contextLength": 50,
# MAGIC   "prefix": ["birth"],
# MAGIC   "suffix": ["faster", "rates"]
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC `OUTPUT: [boy, girl]`

# COMMAND ----------

# MAGIC %md
# MAGIC The parser now takes into account either the prefix OR suffix, only one of the condition has to be fulfilled for a match to count.

# COMMAND ----------

# MAGIC %md
# MAGIC If you remember, the annotator has a `setCompleteContextMatch()` parameter. If you set it to `True`, and used the following JSON configuration :

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Gender",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "girl|boy",
# MAGIC   "contextLength": 50,
# MAGIC   "prefix": ["birth"],
# MAGIC   "suffix": ["fast"]
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC `OUTPUT: []`

# COMMAND ----------

# MAGIC %md
# MAGIC However if we set `setCompleteContextMatch()` as `False`, and use the same JSON configuration as above, we get the following output :

# COMMAND ----------

# MAGIC %md
# MAGIC `OUTPUT: [boy]`

# COMMAND ----------

# MAGIC %md
# MAGIC Here's the sentence again: ***At birth, the typical boy is growing slightly faster than the typical girl, but growth rates become equal at about seven months.***
# MAGIC 
# MAGIC The last 2 properties related to context awareness are `contextException` and `exceptionDistance`. This rules out matches based on a given exception:
# MAGIC   
# MAGIC   ```
# MAGIC {
# MAGIC   "entity": "Gender",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "regex": "girl|boy",
# MAGIC   "contextLength": 50,
# MAGIC   "prefix": ["birth"],
# MAGIC   "suffix": ["faster", "rates"],
# MAGIC   "contextException": ["At"],
# MAGIC   "exceptionDistance": 5
# MAGIC }
# MAGIC ```
# MAGIC `OUTPUT: [girl]`

# COMMAND ----------

# MAGIC %md
# MAGIC Here we've asked the parser to ignore a match if the token "`At`" is within 5 tokens of the matched regex. This caused the token "`boy`" to be ignored.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Dictionary
# MAGIC 
# MAGIC Another key feature of the `ContextualParser` annotator is the use of dictionaries. You can specify a path to a dictionary in `tsv` or `csv` format using the `setDictionary()` parameter. Using a dictionary is a useful when you have a list of exact words that you want the parser to pick up when processing some text.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Orientation
# MAGIC 
# MAGIC The first feature to be aware of when it comes to feeding dictionaries is the format of the dictionaries. The `ContextualParser` annotator will accept dictionaries in the horizontal format and in a vertical format. This is how they would look in practice:

# COMMAND ----------

# MAGIC %md
# MAGIC Horizontal:
# MAGIC 
# MAGIC | normalize | word1 | word2 | word3     |
# MAGIC |-----------|-------|-------|-----------|
# MAGIC | female    | woman | girl  | lady      |
# MAGIC | male      | man   | boy   | gentleman |

# COMMAND ----------

# MAGIC %md
# MAGIC Vertical:
# MAGIC 
# MAGIC | female    | normalize |
# MAGIC |-----------|-----------|
# MAGIC | woman     | word1     |
# MAGIC | girl      | word2     |
# MAGIC | lady      | word3     |

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, your dictionary needs to have a `normalize` field that lets the annotator know which entity labels to use, and another field that lets the annotator know a list of words it should be looking to match. Here's how to set the format that your dictionary uses:
# MAGIC 
# MAGIC ```
# MAGIC contextualParser = ContextualParserApproach() \
# MAGIC     .setDictionary("dictionary.tsv", options={"orientation":"vertical"}) # default is horizontal
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Dictionary-related JSON Properties
# MAGIC 
# MAGIC When working with dictionaries, there are 2 properties in the JSON configuration file to be aware of:
# MAGIC 
# MAGIC - `ruleScope`
# MAGIC - `matchScope`
# MAGIC 
# MAGIC This is especially true when you have multi-word entities in your dictionary.
# MAGIC 
# MAGIC Let's take an example of a dictionary that contains a list of cities, sometimes made up of multiple words:
# MAGIC 
# MAGIC | normalize | word1 | word2 | word3     |
# MAGIC |-----------|-------|-------|-----------|
# MAGIC | City      | New York | Salt Lake City  | Washington      |

# COMMAND ----------

# MAGIC %md
# MAGIC Let's say we're working with the following text: ***I love New York. Salt Lake City is nice too.***
# MAGIC 
# MAGIC With the following JSON properties, here's what you would get:
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "City",
# MAGIC   "ruleScope": "sentence",
# MAGIC   "matchScope": "sub-token",
# MAGIC }
# MAGIC ```
# MAGIC `OUTPUT: []`

# COMMAND ----------

# MAGIC %md
# MAGIC When `ruleScope` is set to `"sentence"`, the annotator attempts to find matches at the token level, parsing through each token in the sentence one by one, looking for a match with the dictionary items. Since `"New York"` and `"Salt Lake City"` are made up of multiple tokens, the annotator would never find a match from the dictionary. Let's change `ruleScope` to `"document"`:
# MAGIC 
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "City",
# MAGIC   "ruleScope": "document",
# MAGIC   "matchScope": "sub-token",
# MAGIC }
# MAGIC ```
# MAGIC `OUTPUT: [New York, Salt Lake City]`

# COMMAND ----------

# MAGIC %md
# MAGIC When `ruleScope` is set to `"document"`, the annotator attempts to find matches by parsing through each sentence in the document one by one, looking for a match with the dictionary items. Beware of how you set `matchScope`. Taking the previous example, if we were to set `matchScope` to `"token"` instead of `"sub-token"`, here's what would happen:
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "City",
# MAGIC   "ruleScope": "document",
# MAGIC   "matchScope": "token"
# MAGIC }
# MAGIC ```
# MAGIC `OUTPUT: [I love New York., Salt Lake City is nice too.]`

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, when `ruleScope` is at the document level, if you set your `matchScope` to the token level, the annotator will output each sentence containing the matched entities as individual chunks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Working with Multi-Word Matches
# MAGIC 
# MAGIC Although not directly related to dictionaries, if we build on top of what we've just seen, there is a use-case that is particularly in demand when working with the `ContextualParser` annotator: finding regex matches for chunks of words that span across multiple tokens. 
# MAGIC 
# MAGIC Let's re-iterate how the `ruleScope` property works: when `ruleScope` is set to `"sentence"`, we're looking for a match on each token of a sentence. When `ruleScope` is set to `"document"`, we're looking for a match on each sentence of a document. 
# MAGIC 
# MAGIC So now let's imagine you're parsing through medical documents trying to tag the *Family History* headers in those documents.
# MAGIC 
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Family History Header",
# MAGIC   "regex": "[f|F]amily\s+[h|H]istory",  
# MAGIC   "ruleScope": "document",
# MAGIC   "matchScope": "sub-token"
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC `OUTPUT: [Family History, family history, Family history]`

# COMMAND ----------

# MAGIC %md
# MAGIC If you had set `ruleScope` to  `"sentence"`, here's what would have happened:
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "entity": "Family History Header",
# MAGIC   "regex": "[f|F]amily\s+[h|H]istory",  
# MAGIC   "ruleScope": "sentence",
# MAGIC   "matchScope": "sub-token"
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC `OUTPUT: []`

# COMMAND ----------

# MAGIC %md
# MAGIC Since Family History is divided into two different tokens, the annotator will never find a match since it's now looking for a match on each token of a sentence.

# COMMAND ----------

# MAGIC %md
# MAGIC # Running a Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1: Detecting Cities
# MAGIC 
# MAGIC Let's try running through some examples to build on top of what you've learned so far.

# COMMAND ----------

# Here's some sample text
sample_text = """Peter Parker is a nice guy and lives in New York . Bruce Wayne is also a nice guy and lives in San Antonio and Gotham City ."""

# COMMAND ----------

# MAGIC %fs mkdirs file:/dbfs/data

# COMMAND ----------

# Create a dictionary to detect cities
cities = """City\nNew York\nGotham City\nSan Antonio\nSalt Lake City"""

with open('/dbfs/data/cities.tsv', 'w') as f:
    f.write(cities)

# Check what dictionary looks like
!cat cities.tsv

# COMMAND ----------

# Create JSON file
cities = {
  "entity": "City",
  "ruleScope": "document", 
  "matchScope":"sub-token",
  "completeMatchRegex": "false"
} 

import json
with open('/dbfs/data/cities.json', 'w') as f:
    json.dump(cities, f)

# COMMAND ----------

# Build pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("entity")\
    .setJsonPath("/dbfs/data/cities.json")\
    .setCaseSensitive(True)\
    .setDictionary('file:/dbfs/data/cities.tsv', options={"orientation":"vertical"})

chunk_converter = ChunkConverter() \
    .setInputCols(["entity"]) \
    .setOutputCol("ner_chunk")

parserPipeline = Pipeline(stages=[
        document_assembler, 
        sentence_detector,
        tokenizer,
        contextual_parser,
        chunk_converter,
        ])

# COMMAND ----------

# Create a lightpipeline model
empty_data = spark.createDataFrame([[""]]).toDF("text")

parserModel = parserPipeline.fit(empty_data)

light_model = LightPipeline(parserModel)

# COMMAND ----------

# Annotate the sample text
annotations = light_model.fullAnnotate(sample_text)[0]

# COMMAND ----------

# Check outputs
annotations.get('ner_chunk')

# COMMAND ----------

# Visualize outputs
from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

ner_vis = visualiser.display(annotations, label_col='ner_chunk', document_col='document', save_path="display_result.html", return_html=True )

displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC Feel free to experiment with the annotator parameters and JSON properties to see how the output might change.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 2: Detect Gender and Age

# COMMAND ----------

# Here's some sample text
sample_text = """A 28 year old female with a history of gestational diabetes mellitus diagnosed 8 years ago. 
                 3 years ago, he reported an episode of HTG-induced pancreatitis . 
                 5 months old boy with repeated concussions."""

# COMMAND ----------

gender = '''male,man,male,boy,gentleman,he,him
female,woman,female,girl,lady,old-lady,she,her
neutral,they,neutral,it'''

with open('/dbfs/data/gender.csv', 'w') as f:
    f.write(gender)


gender = {
  "entity": "Gender",
  "ruleScope": "sentence", 
  "completeMatchRegex": "true",
  "matchScope":"token"
}

import json

with open('/dbfs/data/gender.json', 'w') as f:
    json.dump(gender, f)


age = {
  "entity": "Age",
  "ruleScope": "sentence",
  "matchScope":"token",
  "regex":"\\d{1,3}",
  "prefix":["age of", "age"],
  "suffix": ["-years-old", "years-old", "-year-old",
             "-months-old", "-month-old", "-months-old",
             "-day-old", "-days-old", "month old",
             "days old", "year old", "years old", 
             "years", "year", "months", "old"],
  "contextLength": 25,
  "contextException": ["ago"],
  "exceptionDistance": 12
}

with open('/dbfs/data/age.json', 'w') as f:
    json.dump(age, f)

# COMMAND ----------

# Build pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

gender_contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("chunk_gender") \
    .setJsonPath("/dbfs/data/gender.json") \
    .setCaseSensitive(False) \
    .setDictionary('dbfs:/data/gender.csv', options={"delimiter":","}) \
    .setPrefixAndSuffixMatch(False)      

age_contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("chunk_age") \
    .setJsonPath("/dbfs/data/age.json") \
    .setCaseSensitive(False) \
    .setPrefixAndSuffixMatch(False)\
    .setShortestContextMatch(True)\
    .setOptionalContextRules(False)    

chunk_merger = ChunkMergeApproach() \
    .setInputCols(["chunk_gender", "chunk_age"]) \
    .setOutputCol("ner_chunk")

parserPipeline = Pipeline(stages=[
        document_assembler, 
        sentence_detector,
        tokenizer,
        gender_contextual_parser,
        age_contextual_parser,
        chunk_merger
        ])

# COMMAND ----------

empty_data = spark.createDataFrame([[""]]).toDF("text")

parserModel = parserPipeline.fit(empty_data)

light_model = LightPipeline(parserModel)

# COMMAND ----------

annotations = light_model.fullAnnotate(sample_text)[0]

# COMMAND ----------

annotations.get('ner_chunk')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Highlight the Entities

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualiser = NerVisualizer()

ner_vis = visualiser.display(annotations, label_col='ner_chunk', document_col='document', return_html=True)
  
displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC Feel free to experiment with the annotator parameters and JSON properties to see how the output might change. If you're looking to work on running the pipeline on a full dataset, just make sure to use the `fit()` and `transform()` methods directly on your dataset instead of using the lightpipeline.

# COMMAND ----------

# Create example dataframe with sample text
data = spark.createDataFrame([[sample_text]]).toDF("text")

# Fit and show
results = parserPipeline.fit(data).transform(data)
results.show()

# COMMAND ----------

results.select("chunk_age.result").show()

# COMMAND ----------

results.select("chunk_gender.result").show()