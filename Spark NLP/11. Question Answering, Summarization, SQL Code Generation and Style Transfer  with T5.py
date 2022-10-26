# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 11. Question Answering, Summarization, SQL Code Generation and Style Transfer with T5

# COMMAND ----------

# MAGIC %md
# MAGIC # Question Answering and Summarization with [T5](https://arxiv.org/pdf/1910.10683.pdf)
# MAGIC 
# MAGIC 
# MAGIC Google's T5 is a Sequence to Sequence model that was trained on over 15 different NLP datasets with various problem types, raning from Text Summarization, Question Answering, Translation to various semantical deduction tasks, which enriches T5's ability to map token sequences to semantic vectors which contain more meaning, which T5 leverages to generalize across various tasks and even to never before trained tasks.
# MAGIC 
# MAGIC On top of this, T5 is trained on the standard Word prediction task, which most transformer based models like BERT, GPT, ELMO have been trained on. This gives T5 general knowledge of real world concepts to additionally enhance its understanding.
# MAGIC 
# MAGIC With T5 you can answer **general knowledge based questions given no context** and in addition answer **questions on text databases**.      
# MAGIC These questions can be asked in natural human.
# MAGIC 
# MAGIC 
# MAGIC ## What is a `open book question`? 
# MAGIC You can imagine an `open book` question similar to an examen where you are allowed to bring in text documents or cheat sheets that help you answer questions in an examen. Kinda like bringing a history book to an history examen. 
# MAGIC 
# MAGIC In `T5's` terms, this means the model is given a `question` and an **additional piece of textual information** or so called `context`.
# MAGIC 
# MAGIC This enables the `T5` model to answer questions on textual datasets like `medical records`,`newsarticles` , `wiki-databases` , `stories` and `movie scripts` , `product descriptions`, 'legal documents' and many more.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## What is a `closed book question`? 
# MAGIC A `closed book question` is the exact opposite of a `open book question`. In an examen scenario, you are only allowed to use what you have memorized in your brain and nothing else.      
# MAGIC In `T5's` terms this means that T5 can only use it's stored weights to answer a `question` and is given **no aditional context**.        
# MAGIC `T5` was pre-trained on the [C4 dataset](https://commoncrawl.org/) which contains **petabytes  of web crawling data**  collected over the last 8 years, including Wikipedia in every language.
# MAGIC 
# MAGIC 
# MAGIC This gives `T5` the broad knowledge of the internet stored in it's weights to answer various `closed book questions` 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <!-- [T5]() -->
# MAGIC ![T5 GIF](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s1600/image3.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC <center><h4><b>MODEL LIST</b></h4>
# MAGIC 
# MAGIC Below is a list of Text Generation models. You can get detailed information about the models by clicking on the links.
# MAGIC 
# MAGIC |index|model|lang|
# MAGIC |-----:|:-----|----|
# MAGIC | 1| [t5_active_to_passive_styletransfer](https://nlp.johnsnowlabs.com/2022/05/31/t5_active_to_passive_styletransfer_en_3_0.html)  |en|
# MAGIC | 2| [t5_base](https://nlp.johnsnowlabs.com/2022/05/31/t5_base_en_3_0.html)  |en|
# MAGIC | 3| [t5_base_mediqa_mnli](https://nlp.johnsnowlabs.com/2021/02/19/t5_base_mediqa_mnli_en.html)  |en|
# MAGIC | 4| [t5_formal_to_informal_styletransfer](https://nlp.johnsnowlabs.com/2022/05/31/t5_formal_to_informal_styletransfer_en_3_0.html)  |en|
# MAGIC | 5| [t5_grammar_error_corrector](https://nlp.johnsnowlabs.com/2022/01/12/t5_grammar_error_corrector_en.html)  |en|
# MAGIC | 6| [t5_informal_to_formal_styletransfer](https://nlp.johnsnowlabs.com/2022/05/31/t5_informal_to_formal_styletransfer_en_3_0.html)  |en|
# MAGIC | 7| [t5_passive_to_active_styletransfer](https://nlp.johnsnowlabs.com/2022/05/31/t5_passive_to_active_styletransfer_en_3_0.html)  |en|
# MAGIC | 8| [t5_question_generation_small](https://nlp.johnsnowlabs.com/2022/07/05/t5_question_generation_small_en_3_0.html)  |en|
# MAGIC | 9| [t5_small](https://nlp.johnsnowlabs.com/2022/05/31/t5_small_en_3_0.html)  |en|
# MAGIC | 10| [t5_small_wikiSQL](https://nlp.johnsnowlabs.com/2022/05/31/t5_small_wikiSQL_en_3_0.html)  |en|
# MAGIC 
# MAGIC </center>

# COMMAND ----------

import sparknlp

from sparknlp.base import *
from sparknlp.annotator import *

from pyspark.ml import Pipeline

print("Spark NLP version", sparknlp.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC # Download T5 Model and Create Spark NLP Pipeline

# COMMAND ----------

from sparknlp.common import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document") 

# Can take in document or sentence columns
t5 = T5Transformer.pretrained(name='t5_base',lang='en')\
    .setInputCols('document')\
    .setOutputCol("T5")\
    .setMaxOutputLength(400)

# COMMAND ----------

# MAGIC %md
# MAGIC # Answering Questions

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set the Task to `question`

# COMMAND ----------

# Set the task for questions on T5. Depending to what this is currently set, we get different behaivour
t5.setTask('question')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Answer **Closed Book Questions**  
# MAGIC Closed book means that no additional context is given and the model must answer the question with the knowledge stored in it's weights

# COMMAND ----------

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data
data = [["Who is president of Nigeria? "],
        ["What is the most common language in India? "],
        ["What is the capital of Germany? "],]
df=spark.createDataFrame(data).toDF('text')

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Answer **Open Book Questions** 
# MAGIC These are questions where we give the model some additional context, that is used to answer the question

# COMMAND ----------

from pyspark.sql import SparkSession

context   = 'context: Peters last week was terrible! He had an accident and broke his leg while skiing!'
question1  = 'question: Why was peters week so bad? ' #
question2  = 'question: How did peter broke his leg? ' 
question3  = 'question: How did peter broke his leg? ' 
data = [[question1+context],[question2+context],[question3+context],]
df=spark.createDataFrame(data).toDF('text')

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# Ask T5 questions in the context of a News Article
question1 = 'question: Who is Jack ma? '
question2 = 'question: Who is founder of Alibaba Group? '
question3 = 'question: When did Jack Ma re-appear? '
question4 = 'question: How did Alibaba stocks react? '
question5 = 'question: Whom did Jack Ma meet? '
question6 = 'question: Who did Jack Ma hide from? '


# from https://www.bbc.com/news/business-55728338 
news_article_context = """ context:
Alibaba Group founder Jack Ma has made his first appearance since Chinese regulators cracked down on his business empire.
His absence had fuelled speculation over his whereabouts amid increasing official scrutiny of his businesses.
The billionaire met 100 rural teachers in China via a video meeting on Wednesday, according to local government media.
Alibaba shares surged 5% on Hong Kong's stock exchange on the news.
"""

data = [
             [question1+ news_article_context],
             [question2+ news_article_context],
             [question3+ news_article_context],
             [question4+ news_article_context],
             [question5+ news_article_context],
             [question6+ news_article_context]]


df=spark.createDataFrame(data).toDF('text')

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Summarize documents

# COMMAND ----------

# Set the task for questions on T5
t5.setTask('summarize')

# COMMAND ----------

# https://www.reuters.com/article/instant-article/idCAKBN2AA2WF
text = """(Reuters) - Mastercard Inc said on Wednesday it was planning to offer support for some cryptocurrencies on its network this year, joining a string of big-ticket firms that have pledged similar support.

The credit-card giant’s announcement comes days after Elon Musk’s Tesla Inc revealed it had purchased $1.5 billion of bitcoin and would soon accept it as a form of payment.

Asset manager BlackRock Inc and payments companies Square and PayPal have also recently backed cryptocurrencies.

Mastercard already offers customers cards that allow people to transact using their cryptocurrencies, although without going through its network.

"Doing this work will create a lot more possibilities for shoppers and merchants, allowing them to transact in an entirely new form of payment. This change may open merchants up to new customers who are already flocking to digital assets," Mastercard said. (mstr.cd/3tLaPZM)

Mastercard specified that not all cryptocurrencies will be supported on its network, adding that many of the hundreds of digital assets in circulation still need to tighten their compliance measures.

Many cryptocurrencies have struggled to win the trust of mainstream investors and the general public due to their speculative nature and potential for money laundering.
"""
data = [[text]]
df=spark.createDataFrame(data).toDF('text')
#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['t5.result']).show(truncate=False)

# COMMAND ----------

v = annotated_df.take(1)
print(f"Original Length {len(v[0].text)}   Summarized Length : {len(v[0].T5[0].result)} ")


# COMMAND ----------

# Full summarized text
v[0].T5[0].result

# COMMAND ----------

# MAGIC %md
# MAGIC # SQL Code Generation and Style Transfer with T5

# COMMAND ----------

# MAGIC %md
# MAGIC Google's T5 is a Sequence to Sequence model that was trained on over 15 different NLP datasets with various problem types, raning from Text Summarization, Question Answering, Translation to various semantical deduction tasks, which enriches T5's ability to map token sequences to semantic vectors which contain more meaning, which T5 leverages to generalize across various tasks and even to never before trained tasks.
# MAGIC 
# MAGIC On top of this, T5 is trained on the standard Word prediction task, which most transformer based models like BERT, GPT, ELMO have been trained on. This gives T5 general knowledge of real world concepts to additionally enhance its understanding.

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5-small fine-tuned on WikiSQL
# MAGIC 
# MAGIC Google’s T5 small fine-tuned on WikiSQL for English to SQL translation. Will generate SQL code from natural language input when task is set it to “translate English to SQL:”.

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = T5Transformer.pretrained("t5_small_wikiSQL") \
    .setTask("translate English to SQL:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("sql")

pipeline = Pipeline().setStages([documentAssembler, t5])

data = spark.createDataFrame([["How many customers have ordered more than 2 items?"]]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("sql.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Style Transfer with T5

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 for Active to Passive Style Transfer
# MAGIC 
# MAGIC This is a text-to-text model based on T5 fine-tuned to generate actively written text from a passively written text input, for the task “transfer Active to Passive:”. It is based on Prithiviraj Damodaran’s Styleformer.

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = T5Transformer.pretrained("t5_active_to_passive_styletransfer") \
    .setTask("transfer Active to Passive:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("transfers")

pipeline = Pipeline().setStages([documentAssembler, t5])

data = spark.createDataFrame([["I am writing you a letter."]]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("transfers.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 for Passive to Active Style Transfer
# MAGIC 
# MAGIC This is a text-to-text model based on T5 fine-tuned to generate passively written text from a actively written text input, for the task “transfer Passive to Active:”. It is based on Prithiviraj Damodaran’s Styleformer.

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = T5Transformer.pretrained("t5_passive_to_active_styletransfer") \
    .setTask("transfer Passive to Active:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("transfers")

pipeline = Pipeline().setStages([documentAssembler, t5])

data = spark.createDataFrame([["A letter was sent to you."]]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("transfers.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 for Formal to Informal Style Transfer
# MAGIC 
# MAGIC This is a text-to-text model based on T5 fine-tuned to generate informal text from a formal text input, for the task “transfer Formal to Casual:”. It is based on Prithiviraj Damodaran’s Styleformer.

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = T5Transformer.pretrained("t5_formal_to_informal_styletransfer") \
    .setTask("transfer Formal to Casual:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("transfers")

pipeline = Pipeline().setStages([documentAssembler, t5])

data = spark.createDataFrame([["Please leave the room now."]]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("transfers.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 for Informal to Formal Style Transfer
# MAGIC 
# MAGIC This is a text-to-text model based on T5 fine-tuned to generate informal text from a formal text input, for the task “transfer Casual to Formal:”. It is based on Prithiviraj Damodaran’s Styleformer.

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

t5 = T5Transformer.pretrained("t5_informal_to_formal_styletransfer") \
    .setTask("transfer Casual to Formal:") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(200) \
    .setOutputCol("transfers")

pipeline = Pipeline().setStages([documentAssembler, t5])

data = spark.createDataFrame([["Who gives a crap?"]]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("transfers.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Multi Problem T5 model for Summarization and more
# MAGIC The main T5 model was trained for over 20 tasks from the SQUAD/GLUE/SUPERGLUE datasets. See [this notebook](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/7_T5_SQUAD_GLUE_SUPER_GLUE_TASKS.ipynb) for a demo of all tasks 
# MAGIC 
# MAGIC 
# MAGIC # Overview of every task available with T5
# MAGIC [The T5 model](https://arxiv.org/pdf/1910.10683.pdf) is trained on various datasets for 17 different tasks which fall into 8 categories.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 1. Text summarization
# MAGIC 2. Question answering
# MAGIC 3. Translation
# MAGIC 4. Sentiment analysis
# MAGIC 5. Natural Language inference
# MAGIC 6. Coreference resolution
# MAGIC 7. Sentence Completion
# MAGIC 8. Word sense disambiguation
# MAGIC 
# MAGIC ### Every T5 Task with explanation:
# MAGIC |Task Name | Explanation | 
# MAGIC |----------|--------------|
# MAGIC |[1.CoLA](https://nyu-mll.github.io/CoLA/)                   | Classify if a sentence is gramaticaly correct|
# MAGIC |[2.RTE](https://dl.acm.org/doi/10.1007/11736790_9)                    | Classify whether if a statement can be deducted from a sentence|
# MAGIC |[3.MNLI](https://arxiv.org/abs/1704.05426)                   | Classify for a hypothesis and premise whether they contradict or contradict each other or neither of both (3 class).|
# MAGIC |[4.MRPC](https://www.aclweb.org/anthology/I05-5002.pdf)                   | Classify whether a pair of sentences is a re-phrasing of each other (semantically equivalent)|
# MAGIC |[5.QNLI](https://arxiv.org/pdf/1804.07461.pdf)                   | Classify whether the answer to a question can be deducted from an answer candidate.|
# MAGIC |[6.QQP](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)                    | Classify whether a pair of questions is a re-phrasing of each other (semantically equivalent)|
# MAGIC |[7.SST2](https://www.aclweb.org/anthology/D13-1170.pdf)                   | Classify the sentiment of a sentence as positive or negative|
# MAGIC |[8.STSB](https://www.aclweb.org/anthology/S17-2001/)                   | Classify the sentiment of a sentence on a scale from 1 to 5 (21 Sentiment classes)|
# MAGIC |[9.CB](https://ojs.ub.uni-konstanz.de/sub/index.php/sub/article/view/601)                     | Classify for a premise and a hypothesis whether they contradict each other or not (binary).|
# MAGIC |[10.COPA](https://www.aaai.org/ocs/index.php/SSS/SSS11/paper/view/2418/0)                   | Classify for a question, premise, and 2 choices which choice the correct choice is (binary).|
# MAGIC |[11.MultiRc](https://www.aclweb.org/anthology/N18-1023.pdf)                | Classify for a question, a paragraph of text, and an answer candidate, if the answer is correct (binary),|
# MAGIC |[12.WiC](https://arxiv.org/abs/1808.09121)                    | Classify for a pair of sentences and a disambigous word if the word has the same meaning in both sentences.|
# MAGIC |[13.WSC/DPR](https://www.aaai.org/ocs/index.php/KR/KR12/paper/view/4492/0)       | Predict for an ambiguous pronoun in a sentence what it is referring to.  |
# MAGIC |[14.Summarization](https://arxiv.org/abs/1506.03340)          | Summarize text into a shorter representation.|
# MAGIC |[15.SQuAD](https://arxiv.org/abs/1606.05250)                  | Answer a question for a given context.|
# MAGIC |[16.WMT1.](https://arxiv.org/abs/1706.03762)                  | Translate English to German|
# MAGIC |[17.WMT2.](https://arxiv.org/abs/1706.03762)                   | Translate English to French|
# MAGIC |[18.WMT3.](https://arxiv.org/abs/1706.03762)                   | Translate English to Romanian|