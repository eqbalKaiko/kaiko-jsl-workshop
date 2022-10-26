# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Pretrained pipelines for Grammar, NER and Sentiment

# COMMAND ----------

import sparknlp

print("Spark NLP version", sparknlp.version())

print("Apache Spark version:", spark.version)

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Pretrained Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC https://github.com/JohnSnowLabs/spark-nlp-models
# MAGIC 
# MAGIC https://nlp.johnsnowlabs.com/models

# COMMAND ----------

from sparknlp.pretrained import ResourceDownloader
ResourceDownloader.showPublicPipelines(lang="en")

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

testDoc = '''Peter is a very good persn.
My life in Russia is very intersting.
John and Peter are brthers. However they don't support each other that much.
Lucas Nogal Dunbercker is no longer happy. He has a good car though.
Europe is very culture rich. There are huge churches! and big houses!
'''

# COMMAND ----------

testDoc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explain Document DL

# COMMAND ----------

# MAGIC %md
# MAGIC **Stages**
# MAGIC - DocumentAssembler
# MAGIC - SentenceDetector
# MAGIC - Tokenizer
# MAGIC - NER (NER with GloVe 100D embeddings, CoNLL2003 dataset)
# MAGIC - Lemmatizer
# MAGIC - Stemmer
# MAGIC - Part of Speech
# MAGIC - SpellChecker (Norvig)

# COMMAND ----------

pipeline_dl = PretrainedPipeline('explain_document_dl', lang='en')


# COMMAND ----------

pipeline_dl.model.stages

# COMMAND ----------

pipeline_dl.model.stages[-2].getStorageRef()

# COMMAND ----------

pipeline_dl.model.stages[-2].getClasses()

# COMMAND ----------

result = pipeline_dl.annotate(testDoc)

result.keys()

# COMMAND ----------

result['entities']

# COMMAND ----------

import pandas as pd

df = pd.DataFrame({'token':result['token'], 'ner_label':result['ner'],
                      'spell_corrected':result['checked'], 'POS':result['pos'],
                      'lemmas':result['lemma'], 'stems':result['stem']})

df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recognize Entities DL

# COMMAND ----------

recognize_entities = PretrainedPipeline('recognize_entities_dl', lang='en')


# COMMAND ----------

testDoc = '''
Peter is a very good persn.
My life in Russia is very intersting.
John and Peter are brthers. However they don't support each other that much.
Lucas Nogal Dunbercker is no longer happy. He has a good car though.
Europe is very culture rich. There are huge churches! and big houses!
'''

result = recognize_entities.annotate(testDoc)

list(zip(result['token'], result['ner']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean Stop Words

# COMMAND ----------

clean_stop = PretrainedPipeline('clean_stop', lang='en')


# COMMAND ----------

result = clean_stop.annotate(testDoc)
result.keys()

# COMMAND ----------

' '.join(result['cleanTokens'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spell Checker 
# MAGIC 
# MAGIC (Norvig Algo)
# MAGIC 
# MAGIC ref: https://norvig.com/spell-correct.html

# COMMAND ----------

spell_checker = PretrainedPipeline('check_spelling', lang='en')


# COMMAND ----------

testDoc = '''
Peter is a very good persn.
My life in Russia is very intersting.
John and Peter are brthers. However they don't support each other that much.
Lucas Nogal Dunbercker is no longer happy. He has a good car though.
Europe is very culture rich. There are huge churches! and big houses!
'''

result = spell_checker.annotate(testDoc)

result.keys()

# COMMAND ----------

list(zip(result['token'], result['checked']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parsing a list of texts

# COMMAND ----------

testDoc_list = ['French author who helped pioner the science-fiction genre.',
'Verne wrate about space, air, and underwater travel before navigable aircrast',
'Practical submarines were invented, and before any means of space travel had been devised.']

testDoc_list

# COMMAND ----------

pipeline_dl = PretrainedPipeline('explain_document_dl', lang='en')


# COMMAND ----------

result_list = pipeline_dl.annotate(testDoc_list)

len(result_list)

# COMMAND ----------

result_list[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using fullAnnotate to get more details

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC annotatorType: String, 
# MAGIC begin: Int, 
# MAGIC end: Int, 
# MAGIC result: String, (this is what annotate returns)
# MAGIC metadata: Map[String, String], 
# MAGIC embeddings: Array[Float]
# MAGIC ```

# COMMAND ----------

text = 'Peter Parker is a nice guy and lives in New York'

# COMMAND ----------

# pipeline_dl >> explain_document_dl

detailed_result = pipeline_dl.fullAnnotate(text)

# COMMAND ----------

detailed_result

# COMMAND ----------

detailed_result[0]['entities']

# COMMAND ----------

detailed_result[0]['entities'][0].result

# COMMAND ----------

import pandas as pd

chunks=[]
entities=[]
for n in detailed_result[0]['entities']:
        
  chunks.append(n.result)
  entities.append(n.metadata['entity']) 
    
df = pd.DataFrame({'chunks':chunks, 'entities':entities})
df    

# COMMAND ----------

tuples = []

for x,y,z in zip(detailed_result[0]["token"], detailed_result[0]["pos"], detailed_result[0]["ner"]):

  tuples.append((int(x.metadata['sentence']), x.result, x.begin, x.end, y.result, z.result))

df = pd.DataFrame(tuples, columns=['sent_id','token','start','end','pos', 'ner'])

df


# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentiment Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vivek algo
# MAGIC 
# MAGIC paper: `Fast and accurate sentiment classification using an enhanced Naive Bayes model`
# MAGIC 
# MAGIC https://arxiv.org/abs/1305.6143
# MAGIC 
# MAGIC code `https://github.com/vivekn/sentiment`

# COMMAND ----------

sentiment = PretrainedPipeline('analyze_sentiment', lang='en')

# COMMAND ----------

result = sentiment.annotate("The movie I watched today was not a good one")

result['sentiment']

# COMMAND ----------

# MAGIC %md
# MAGIC #### DL version (trained on imdb)

# COMMAND ----------

# MAGIC %md
# MAGIC `analyze_sentimentdl_use_imdb`: A pre-trained pipeline to classify IMDB reviews in neg and pos classes using tfhub_use embeddings.
# MAGIC 
# MAGIC `analyze_sentimentdl_glove_imdb`: A pre-trained pipeline to classify IMDB reviews in neg and pos classes using glove_100d embeddings.

# COMMAND ----------

sentiment_imdb_glove = PretrainedPipeline('analyze_sentimentdl_glove_imdb', lang='en')

# COMMAND ----------

comment = '''
It's a very scary film but what impressed me was how true the film sticks to the original's tricks; it isn't filled with loud in-your-face jump scares, in fact, a lot of what makes this film scary is the slick cinematography and intricate shadow play. The use of lighting and creation of atmosphere is what makes this film so tense, which is why it's perfectly suited for those who like Horror movies but without the obnoxious gore.
'''
result = sentiment_imdb_glove.annotate(comment)

result['sentiment']

# COMMAND ----------

sentiment_imdb_glove.fullAnnotate(comment)[0]['sentiment']

# COMMAND ----------

# MAGIC %md
# MAGIC #### DL version (trained on twitter dataset)

# COMMAND ----------

sentiment_twitter = PretrainedPipeline('analyze_sentimentdl_use_twitter', lang='en')

# COMMAND ----------

result = sentiment_twitter.annotate("The movie I watched today was a good one.")

result['sentiment']

# COMMAND ----------

sentiment_twitter.fullAnnotate("The movie I watched today was a good one.")[0]['sentiment']

# COMMAND ----------

# MAGIC %md
# MAGIC End of Notebook #