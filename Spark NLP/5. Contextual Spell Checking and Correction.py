# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Contextual Spell Checking and Correction

# COMMAND ----------

# MAGIC %md
# MAGIC <H1> Noisy Channel Model Spell Checker - Introduction </H1>
# MAGIC 
# MAGIC blogpost : https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc
# MAGIC 
# MAGIC <div>
# MAGIC <p><br/>
# MAGIC The idea for this annotator is to have a flexible, configurable and "re-usable by parts" model.<br/>
# MAGIC Flexibility is the ability to accommodate different use cases for spell checking like OCR text, keyboard-input text, ASR text, and general spelling problems due to orthographic errors.<br/>
# MAGIC We say this is a configurable annotator, as you can adapt it yourself to different use cases avoiding re-training as much as possible.<br/>
# MAGIC </p>
# MAGIC </div>
# MAGIC 
# MAGIC 
# MAGIC <b> Spell Checking at three levels: </b>
# MAGIC The final ranking of a correction sequence is affected by three things, 
# MAGIC 
# MAGIC 
# MAGIC 1. Different correction candidates for each word - __word level__.
# MAGIC 2. The surrounding text of each word, i.e. it's context - __sentence level__.
# MAGIC 3. The relative cost of different correction candidates according to the edit operations at the character level it requires - __subword level__.

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

print("Spark NLP version", sparknlp.version())

print("Apache Spark version:", spark.version)

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initial Setup
# MAGIC As it's usual in Spark-NLP let's start with building a pipeline; a _spell correction pipeline_. We will use a pretrained model from our library.

# COMMAND ----------

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

from IPython.utils.text import columnize

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = RecursiveTokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")\
    .setPrefixes(["\"", "(", "[", "\n"])\
    .setSuffixes([".", ",", "?", ")","!", "'s"])

spellModel = ContextSpellCheckerModel\
    .pretrained('spellcheck_dl')\
    .setInputCols("token")\
    .setOutputCol("checked")\
    .setErrorThreshold(4.0)\
    .setTradeoff(6.0)

finisher = Finisher()\
    .setInputCols("checked")

pipeline = Pipeline(
    stages = [
    documentAssembler,
    tokenizer,
    spellModel,
    finisher
  ])

empty_ds = spark.createDataFrame([[""]]).toDF("text")
lp = LightPipeline(pipeline.fit(empty_ds))

# COMMAND ----------

# MAGIC %md
# MAGIC Ok!, at this point we have our spell checking pipeline as expected. Let's see what we can do with it,

# COMMAND ----------

lp.annotate("Plaese alliow me tao introdduce myhelf, I am a man of waelth und tiaste")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Word Level Corrections
# MAGIC Continuing with our pretrained model, let's try to see how corrections work at the word level. Each Context Spell Checker model that you can find in Spark-NLP library comes with two sources for word candidates: 
# MAGIC + a general vocabulary that is built during training(and remains unmutable during the life of the model), and
# MAGIC + special classes for dealing with special types of words like numbers or dates. These are dynamic, and you can modify them so they adjust better to your data.
# MAGIC 
# MAGIC The general vocabulary is learned during training, and cannot be modified, however, the special classes can be updated after training has happened on a pre-trained model.
# MAGIC This means you can modify how existing classes produce corrections, but not the number or type of the classes.
# MAGIC Let's see how we can accomplish this.

# COMMAND ----------

# First let's start with a loaded model, and check which classes it has been trained with
spellModel.getWordClasses()

# COMMAND ----------

# MAGIC %md
# MAGIC We have five classes, of two different types: some are vocabulary based and others are regex based,
# MAGIC + __Vocabulary based classes__ can propose correction candidates from the provided vocabulary, for example a dictionary of names.
# MAGIC + __Regex classes__ are defined by a regular expression, and they can be used to generate correction candidates for things like numbers. Internally, the Spell Checker will enumerate your regular expression and build a fast automaton, not only for recognizing the word(number in this example) as valid and preserve it, but also for generating a correction candidate.
# MAGIC Thus the regex should be a finite regex(it must define a finite regular language).
# MAGIC 
# MAGIC Now suppose that you have a new friend from Poland whose name is 'Jowita', let's see how the pretrained Spell Checker does with this name.

# COMMAND ----------

beautify = lambda annotations: [columnize(sent['checked']) for sent in annotations]

# COMMAND ----------

# Foreign name without errors
sample = 'We are going to meet Jowita in the city hall.'
beautify([lp.annotate(sample)])

# COMMAND ----------

# MAGIC %md
# MAGIC Well, the result is not very good, that's because the Spell Checker has been trained mainly with American English texts. At least, the surrounding words are helping to obtain a correction that is a name. We can do better, let's see how.
# MAGIC 
# MAGIC ## Updating a predefined word class
# MAGIC 
# MAGIC ### Vocabulary Classes
# MAGIC 
# MAGIC In order for the Spell Checker to be able to preserve words, like a foreign name, we have the option to update existing classes so they can cover new words.

# COMMAND ----------

# add some more, in case we need them
spellModel.updateVocabClass('_NAME_', ['Monika', 'Agnieszka', 'Inga', 'Jowita', 'Melania'], True)

# Let's see what we get now
sample = 'We are going to meet Jowita at the city hall.'
beautify([lp.annotate(sample)])

# COMMAND ----------

# MAGIC %md
# MAGIC Much better, right? Now suppose that we want to be able to not only preserve the word, but also to propose meaningful corrections to the name of our foreign friend.

# COMMAND ----------

# Foreign name with an error
sample = 'We are going to meet Jovita in the city hall.'
beautify([lp.annotate(sample)])

# COMMAND ----------

# MAGIC %md
# MAGIC Here we were able to add the new word to the class and propose corrections for it, but also, the new word has been treated as a name, that meaning that the model used information about the typical context for names in order to produce the best correction.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regex Classes
# MAGIC We can do something similar for classes defined by regex. We can add a regex, to for example deal with a special format for dates, that will not only preserve the date with the special format, but also be able to correct it.

# COMMAND ----------

# Date with custom format
sample = 'We are going to meet her in the city hall on february-3.'
beautify([lp.annotate(sample)])

# COMMAND ----------

# this is a sample regex, for simplicity not covering all months
spellModel.updateRegexClass('_DATE_', '(january|february|march)-[0-31]')
beautify([lp.annotate(sample)])

# COMMAND ----------

# MAGIC %md
# MAGIC Now our date wasn't destroyed!

# COMMAND ----------

# now check that it produces good corrections to the date
sample = 'We are going to meet her in the city hall on mebbruary-3.'
beautify([lp.annotate(sample)])

# COMMAND ----------

# MAGIC %md
# MAGIC And the model produces good corrections for the special regex class. Remember that each regex that you enter to the model must be finite. In all these examples the new definitions for our classes didn't prevent the model to continue using the context to produce corrections. Let's see why being able to use the context is important.
# MAGIC ### Sentence Level Corrections
# MAGIC The Spell Checker can leverage the context of words for ranking different correction sequences. Let's take a look at some examples,

# COMMAND ----------

# check for the different occurrences of the word "siter"
example1 = ["I will call my siter.",\
            "Due to bad weather, we had to move to a different siter.",\
            "We travelled to three siter in the summer."]
beautify(lp.annotate(example1))

# COMMAND ----------

# check for the different occurrences of the word "ueather"
example2 = ["During the summer we have the best ueather.",\
            "I have a black ueather jacket, so nice."]
beautify(lp.annotate(example2))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that in the first example, 'siter' is indeed a valid English word, <br/> https://www.merriam-webster.com/dictionary/siter <br/>
# MAGIC The only way to customize how the use of context is performed is to train the language model by training a Spell Checker from scratch. If you want to be able to train your custom language model, please refer to the Training notebook.
# MAGIC Now we've learned how the context can help to pick the best possible correction, and why it is important to be able to leverage the context even when the other parts of the Spell Checker were updated.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subword level corrections
# MAGIC Another fine tunning that our Spell Checker accepts is to assign different costs to different edit operations that are necessary to transform a word into a correction candidate. 
# MAGIC So, why is this important? Errors can come from different sources,
# MAGIC + Homophones are words that sound similar, but are written differently and have different meaning. Some examples, {there, their, they're}, {see, sea}, {to, too, two}. You will typically see these errors in text obtained by Automatic Speech Recognition(ASR).
# MAGIC + Characters can also be confused because of looking similar. So a 0(zero) can be confused with a O(capital o), or a 1(number one) with an l(lowercase l). These errors typically come from OCR.
# MAGIC + Input device related, sometimes keyboards cause certain patterns to be more likely than others due to letter locations, for example in a QWERTY keyboard.
# MAGIC + Last but not least, ortographic errors, related to the writter making mistakes. Forgetting a double consonant, or using it in the wrong place, interchanging letters(i.e., 'becuase' for 'because'), and many others.
# MAGIC 
# MAGIC The goal is to continue using all the other features of the model and still be able to adapt the model to handle each of these cases in the best possible way. Let's see how to accomplish this.

# COMMAND ----------

# sending or lending ?
sample = 'I will be 1ending him my car'
lp.annotate(sample)

# COMMAND ----------

# let's make the replacement of an '1' for an 'l' cheaper
weights = {'1': {'l': .01}}
spellModel.setWeights(weights)
lp.annotate(sample)

# COMMAND ----------

# MAGIC %md
# MAGIC Assembling this matrix by hand could be a daunting challenge. There is one script in Python that can do this for you.
# MAGIC This is something to be soon included like an option during training for the Context Spell Checker. Stay tuned on new releases!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced - the mysterious tradeoff parameter 
# MAGIC There's a clear tension between two forces here,
# MAGIC + The context information: by which the model wants to change words based on the surrounding words.
# MAGIC + The word information: by which the model wants to preserve as much an input word as possible to avoid destroying the input.
# MAGIC 
# MAGIC Changing words that are in the vocabulary for others that seem more suitable according to the context is one of the most challenging tasks in spell correction. This is because you run into the risk of destroying existing 'good' words.
# MAGIC The models that you will find in the Spark-NLP library have already been configured in a way that balances these two forces and produces good results in most of the situations. But your dataset can be different from the one used to train the model.
# MAGIC So we encourage the user to play a bit with the hyperparameters, and for you to have an idea on how it can be modified, we're going to see the following example,

# COMMAND ----------

sample = 'have you been two the falls?'
beautify([lp.annotate(sample)])

# COMMAND ----------

# MAGIC %md
# MAGIC Here 'two' is clearly wrong, probably a typo, and the model should be able to choose the right correction candidate according to the context. <br/>
# MAGIC Every path is scored with a cost, and the higher the cost the less chances for the path being chosen as the final answer.<br/>
# MAGIC In order for the model to rely more on the context and less on word information, we have the setTradeoff() method. You can think of the tradeoff as how much a single edition(insert, delete, etc) operation affects the influence of a word when competing inside a path in the graph.<br/>
# MAGIC So the lower the tradeoff, the less we care about the edit operations in the word, and the more we care about the word fitting properly into its context. The tradeoff parameter typically ranges between 5 and 25. <br/>
# MAGIC Let's see what happens when we relax how much the model cares about individual words in our example,

# COMMAND ----------

spellModel.getTradeoff()

# COMMAND ----------

# let's decrease the influence of word-level errors
# TODO a nicer way of doing this other than re-creating the pipeline?
spellModel.setTradeoff(2.0)

pipeline = Pipeline(
    stages = [
    documentAssembler,
    tokenizer,
    spellModel,
    finisher
  ])

empty_ds = spark.createDataFrame([[""]]).toDF("text")
lp = LightPipeline(pipeline.fit(empty_ds))

beautify([lp.annotate(sample)])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced - performance

# COMMAND ----------

# MAGIC %md
# MAGIC The discussion about performance revolves around _error detection_. The more errors the model detects the more populated is the candidate diagram we showed above[TODO add diagram or convert this into blogpost], and the more alternative paths need to be evaluated. </br>
# MAGIC Basically the error detection stage of the model can decide whether a word needs a correction or not; with two reasons for a word to be considered as incorrect, 
# MAGIC + The word is OOV: the word is out of the vocabulary.
# MAGIC + The context: the word doesn't fit well within its neighbouring words. 
# MAGIC The only parameter that we can control at this point is the second one, and we do so with the setErrorThreshold() method that contains a max perplexity above which the word will be considered suspicious and a good candidate for being corrected.</br>
# MAGIC The parameter that comes with the pretrained model has been set so you can get both a decent performance and accuracy. For reference, this is how the F-score, and time varies in a sample dataset for different values of the errorThreshold,
# MAGIC 
# MAGIC 
# MAGIC |fscore |totaltime|threshold|
# MAGIC |-------|---------|---------|
# MAGIC |52.69  |405s | 8f|
# MAGIC |52.43  |357s |10f|
# MAGIC |52.25  |279s |12f|
# MAGIC |52.14  |234s |14f|
# MAGIC 
# MAGIC You can trade some minor points in accuracy for a nice speedup.

# COMMAND ----------

def sparknlp_spell_check(text):

  return beautify([lp.annotate(text)])[0].rstrip()


# COMMAND ----------

sparknlp_spell_check('I will go to Philadelhia tomorrow')

# COMMAND ----------

sparknlp_spell_check('I will go to Philadhelpia tomorrow')

# COMMAND ----------

sparknlp_spell_check('I will go to Piladelphia tomorrow')

# COMMAND ----------

sparknlp_spell_check('I will go to Philadedlhia tomorrow')

# COMMAND ----------

sparknlp_spell_check('I will go to Phieladelphia tomorrow')

# COMMAND ----------

# MAGIC %md
# MAGIC ## ContextSpellCheckerApproach
# MAGIC 
# MAGIC Trains a deep-learning based Noisy Channel Model Spell Algorithm.
# MAGIC 
# MAGIC Correction candidates are extracted combining context information and word information.
# MAGIC 
# MAGIC 1.   Different correction candidates for each word   **word level**
# MAGIC 2.   The surrounding text of each word, i.e. it’s context  **sentence level**.
# MAGIC 3.   The relative cost of different correction candidates according to the edit operations at the character level it requires  **subword level**.

# COMMAND ----------

# For this example, we will use the first Sherlock Holmes book as the training dataset.

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols("document") \
    .setOutputCol("token")

spellChecker = ContextSpellCheckerApproach() \
    .setInputCols("token") \
    .setOutputCol("corrected") \
    .setWordMaxDistance(3) \
    .setBatchSize(24) \
    .setEpochs(8) \
    .setLanguageModelClasses(1650)  # dependant on vocabulary size
    # .addVocabClass("_NAME_", names) # Extra classes for correction could be added like this

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    spellChecker
])

# COMMAND ----------

! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Public/data/holmes.txt
  
dbutils.fs.cp("file:/databricks/driver/holmes.txt", "dbfs:/") 
  

# COMMAND ----------

path = "/holmes.txt"

dataset = spark.read.text(path).toDF("text")

dataset.show(truncate=100)

# COMMAND ----------

pipelineModel = pipeline.fit(dataset)

# COMMAND ----------

lp = LightPipeline(pipelineModel)
result = lp.annotate("Plaese alliow me tao introdduce myhelf, I am a man of waelth und tiaste")
result["corrected"]

# COMMAND ----------

import pandas as pd

pd.DataFrame(zip(result["token"],result["corrected"]),columns=["orginal","corrected"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### End of Notebook # 5