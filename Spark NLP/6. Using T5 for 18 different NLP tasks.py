# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # **6. Using T5 for 18 different NLP tasks **

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Overview of every task available with T5
# MAGIC [The T5 model](https://arxiv.org/pdf/1910.10683.pdf) is trained on various datasets for 17 different tasks which fall into 8 categories.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 1. Text Summarization
# MAGIC 2. Question Answering
# MAGIC 3. Translation
# MAGIC 4. Sentiment analysis
# MAGIC 5. Natural Language Inference
# MAGIC 6. Coreference Resolution
# MAGIC 7. Sentence Completion
# MAGIC 8. Word Sense Disambiguation
# MAGIC 
# MAGIC # Every T5 Task with explanation:
# MAGIC |Task Name | Explanation | 
# MAGIC |----------|--------------|
# MAGIC |[1.CoLA](https://nyu-mll.github.io/CoLA/)                   | Classify if a sentence is gramaticaly correct|
# MAGIC |[2.RTE](https://dl.acm.org/doi/10.1007/11736790_9)                    | Classify whether a statement can be deducted from a sentence|
# MAGIC |[3.MNLI](https://arxiv.org/abs/1704.05426)                   | Classify for a hypothesis and premise whether they contradict or imply each other or neither of both (3 class).|
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
# MAGIC 
# MAGIC 
# MAGIC # Information about pre-procession for T5 tasks
# MAGIC 
# MAGIC ## Tasks that require no pre-processing
# MAGIC The following tasks work fine without any additional pre-processing, only setting the `task parameter` on the T5 model is required:
# MAGIC 
# MAGIC -  CoLA
# MAGIC -  Summarization
# MAGIC -  SST2
# MAGIC -  WMT1.
# MAGIC -  WMT2.
# MAGIC -  WMT3.
# MAGIC 
# MAGIC 
# MAGIC ## Tasks that require pre-processing with 1 tag
# MAGIC The following tasks require `exactly 1 additional tag` added by manual pre-processing.
# MAGIC Set the `task parameter` and then join the sentences on the `tag` for these tasks.
# MAGIC 
# MAGIC - RTE
# MAGIC - MNLI
# MAGIC - MRPC
# MAGIC - QNLI
# MAGIC - QQP
# MAGIC - SST2
# MAGIC - STSB
# MAGIC - CB
# MAGIC 
# MAGIC 
# MAGIC ## Tasks that require pre-processing with multiple tags
# MAGIC The following tasks require `more than 1 additional tag` added manual by pre-processing.
# MAGIC Set the `task parameter` and then prefix sentences with their corresponding tags and join them for these tasks:
# MAGIC 
# MAGIC - COPA
# MAGIC - MultiRc
# MAGIC - WiC
# MAGIC 
# MAGIC 
# MAGIC ## WSC/DPR is a special case that requires `*` surrounding
# MAGIC The task WSC/DPR requires highlighting a pronoun with `*` and configuring a `task parameter`.
# MAGIC <br><br><br><br><br>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC The following sections describe each task in detail, with an example and also a pre-processed example.
# MAGIC 
# MAGIC ***NOTE:***  Linebreaks are added to the `pre-processed examples` in the following section. The T5 model also works with linebreaks, but it can hinder the performance and it is not recommended to intentionally add them.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Task 1 [CoLA - Binary Grammatical Sentence acceptability classification](https://nyu-mll.github.io/CoLA/)
# MAGIC Judges if a sentence is grammatically acceptable.     
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Example
# MAGIC 
# MAGIC |sentence  | prediction|
# MAGIC |------------|------------|
# MAGIC | Anna and Mike is going skiing and they is liked is | unacceptable |      
# MAGIC | Anna and Mike like to dance | acceptable | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for CoLA
# MAGIC `.setTask(cola sentence:)` prefix.
# MAGIC 
# MAGIC ### Example pre-processed input for T5 CoLA sentence acceptability judgement:
# MAGIC ```
# MAGIC cola 
# MAGIC sentence: Anna and Mike is going skiing and they is liked is
# MAGIC ```
# MAGIC 
# MAGIC # Task 2 [RTE - Natural language inference Deduction Classification](https://dl.acm.org/doi/10.1007/11736790_9)
# MAGIC The RTE task is defined as recognizing, given two text fragments, whether the meaning of one text can be inferred (entailed) from the other or not.       
# MAGIC Classification of sentence pairs as entailed and not_entailed       
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and  [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Example
# MAGIC 
# MAGIC |sentence 1 | sentence 2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC Kessler ’s team conducted 60,643 interviews with adults in 14 countries.  |  Kessler ’s team interviewed more than 60,000 adults in 14 countries | entailed
# MAGIC Peter loves New York, it is his favorite city| Peter loves new York. | entailed
# MAGIC Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.  |Johnny is a millionare | entailment|
# MAGIC Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.  |Johnny is a poor man  | not_entailment | 
# MAGIC | It was raining in England for the last 4 weeks | England was very dry yesterday | not_entailment|
# MAGIC 
# MAGIC ## How to configure T5 task for RTE
# MAGIC `.setTask('rte sentence1:)` and prefix second sentence with `sentence2:`
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 RTE - 2 Class Natural language inference
# MAGIC ```
# MAGIC rte 
# MAGIC sentence1: Recent report say Peter makes he alot of money, he earned 10 million USD each year for the last 5 years. 
# MAGIC sentence2: Peter is a millionare.
# MAGIC ```
# MAGIC 
# MAGIC ### References
# MAGIC - https://arxiv.org/abs/2010.03061
# MAGIC 
# MAGIC 
# MAGIC # Task 3 [MNLI - 3 Class Natural Language Inference 3-class contradiction classification](https://arxiv.org/abs/1704.05426)
# MAGIC Classification of sentence pairs with the labels `entailment`, `contradiction`, and `neutral`.      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC This classifier predicts for two sentences :
# MAGIC - Whether the first sentence logically and semantically follows from the second sentence as entailment
# MAGIC - Whether the first sentence is a contradiction to the second sentence as a contradiction
# MAGIC - Whether the first sentence does not entail or contradict the first sentence as neutral
# MAGIC 
# MAGIC | Hypothesis | Premise | prediction|
# MAGIC |------------|------------|----------|
# MAGIC | Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years. |    Johnny is a poor man.  | contradiction|
# MAGIC |It rained in England the last 4 weeks.| It was snowing in New York last week| neutral | 
# MAGIC 
# MAGIC ## How to configure T5 task for MNLI
# MAGIC `.setTask('mnli hypothesis:)` and prefix second sentence with `premise:`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 MNLI - 3 Class Natural Language Inference
# MAGIC 
# MAGIC ```
# MAGIC mnli 
# MAGIC hypothesis: At 8:34, the Boston Center controller received a third, transmission from American 11.    
# MAGIC premise: The Boston Center controller got a third transmission from American 11.
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC # Task 4 [MRPC - Binary Paraphrasing/ sentence similarity classification ](https://www.aclweb.org/anthology/I05-5002.pdf)
# MAGIC Detect whether one sentence is a re-phrasing or similar to another sentence      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC | Sentence1 | Sentence2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .| Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 " . | equivalent | 
# MAGIC | I like to eat peanutbutter for breakfast| I like to play football | not_equivalent | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for MRPC
# MAGIC `.setTask('mrpc sentence1:)` and prefix second sentence with `sentence2:`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 MRPC - Binary Paraphrasing/ sentence similarity
# MAGIC 
# MAGIC ```
# MAGIC mrpc 
# MAGIC sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said . 
# MAGIC sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11",
# MAGIC ```
# MAGIC 
# MAGIC *ISSUE:* Can only get neutral and contradiction as prediction results for tested samples but no entailment predictions.
# MAGIC 
# MAGIC 
# MAGIC # Task 5 [QNLI - Natural Language Inference question answered classification](https://arxiv.org/pdf/1804.07461.pdf)
# MAGIC Classify whether a question is answered by a sentence (`entailed`).       
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC | Question | Answer | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |Where did Jebe die?| Ghenkis Khan recalled Subtai back to Mongolia soon afterward, and Jebe died on the road back to Samarkand | entailment|
# MAGIC |What does Steve like to eat? | Steve watches TV all day | not_netailment
# MAGIC 
# MAGIC ## How to configure T5 task for QNLI - Natural Language Inference question answered classification
# MAGIC `.setTask('QNLI sentence1:)` and prefix question with `question:` sentence with `sentence:`:
# MAGIC 
# MAGIC ### Example pre-processed input for T5 QNLI - Natural Language Inference question answered classification
# MAGIC 
# MAGIC ```
# MAGIC qnli
# MAGIC question: Where did Jebe die?     
# MAGIC sentence: Ghenkis Khan recalled Subtai back to Mongolia soon afterwards, and Jebe died on the road back to Samarkand,
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC # Task 6 [QQP - Binary Question Similarity/Paraphrasing](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
# MAGIC Based on a quora dataset, determine whether a pair of questions are semantically equivalent.      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC | Question1 | Question2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |What attributes would have made you highly desirable in ancient Rome?        | How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER? | not_duplicate | 
# MAGIC |What was it like in Ancient rome?  | What was Ancient rome like?| duplicate | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for QQP
# MAGIC .setTask('qqp question1:) and
# MAGIC prefix second sentence with question2:
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 QQP - Binary Question Similarity/Paraphrasing
# MAGIC 
# MAGIC ```
# MAGIC qqp 
# MAGIC question1: What attributes would have made you highly desirable in ancient Rome?        
# MAGIC question2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
# MAGIC ```
# MAGIC 
# MAGIC # Task 7 [SST2 - Binary Sentiment Analysis](https://www.aclweb.org/anthology/D13-1170.pdf)
# MAGIC Binary sentiment classification.      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC | Sentence1 | Prediction  | 
# MAGIC |-----------|-----------|
# MAGIC |it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight |  positive| 
# MAGIC |I really hated that movie | negative | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for  SST2
# MAGIC `.setTask('sst2 sentence: ')`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 SST2 - Binary Sentiment Analysis
# MAGIC 
# MAGIC ```
# MAGIC sst2
# MAGIC sentence: I hated that movie
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Task8 [STSB - Regressive semantic sentence similarity](https://www.aclweb.org/anthology/S17-2001/)
# MAGIC Measures how similar two sentences are on a scale from 0 to 5 with 21 classes representing a regressive label.     
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC | Question1 | Question2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |What attributes would have made you highly desirable in ancient Rome?        | How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER? | 0 | 
# MAGIC |What was it like in Ancient rome?  | What was Ancient rome like?| 5.0 | 
# MAGIC |What was live like as a King in Ancient Rome??       | What is it like to live in Rome? | 3.2 | 
# MAGIC 
# MAGIC ## How to configure T5 task for STSB
# MAGIC `.setTask('stsb sentence1:)` and prefix second sentence with `sentence2:`
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 STSB - Regressive semantic sentence similarity
# MAGIC 
# MAGIC ```
# MAGIC stsb
# MAGIC sentence1: What attributes would have made you highly desirable in ancient Rome?        
# MAGIC sentence2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC # Task 9[ CB -  Natural language inference contradiction classification](https://ojs.ub.uni-konstanz.de/sub/index.php/sub/article/view/601)
# MAGIC Classify whether a Premise contradicts a Hypothesis.    
# MAGIC Predicts entailment, neutral and contradiction     
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC | Hypothesis | Premise | Prediction | 
# MAGIC |--------|-------------|----------|
# MAGIC |Valence was helping | Valence the void-brain, Valence the virtuous valet. Why couldn’t the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping'| Contradiction|
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for CB
# MAGIC `.setTask('cb hypothesis:)` and prefix premise with `premise:`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 CB -  Natural language inference contradiction classification
# MAGIC 
# MAGIC ```
# MAGIC cb 
# MAGIC hypothesis: Valence was helping      
# MAGIC premise: Valence the void-brain, Valence the virtuous valet. Why couldn’t the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping,
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC # Task 10 [COPA - Sentence Completion/ Binary choice selection](https://www.aaai.org/ocs/index.php/SSS/SSS11/paper/view/2418/0)
# MAGIC The Choice of Plausible Alternatives (COPA) task by Roemmele et al. (2011) evaluates
# MAGIC causal reasoning between events, which requires commonsense knowledge about what usually takes
# MAGIC place in the world. Each example provides a premise and either asks for the correct cause or effect
# MAGIC from two choices, thus testing either ``backward`` or `forward causal reasoning`. COPA data, which
# MAGIC consists of 1,000 examples total, can be downloaded at https://people.ict.usc.e
# MAGIC 
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC This classifier selects from a choice of `2 options` which one the correct is based on a `premise`.
# MAGIC 
# MAGIC 
# MAGIC ## forward causal reasoning
# MAGIC Premise: The man lost his balance on the ladder.     
# MAGIC question: What happened as a result?        
# MAGIC Alternative 1: He fell off the ladder.       
# MAGIC Alternative 2: He climbed up the ladder.
# MAGIC ## backwards causal reasoning
# MAGIC Premise: The man fell unconscious. What was the cause
# MAGIC of this?       
# MAGIC Alternative 1: The assailant struck the man in the head.      
# MAGIC Alternative 2: The assailant took the man’s wallet.
# MAGIC 
# MAGIC 
# MAGIC | Question | Premise | Choice 1 | Choice  2 | Prediction | 
# MAGIC |--------|-------------|----------|---------|-------------|
# MAGIC |effect | Politcal Violence broke out in the nation. | many citizens relocated to the capitol. |  Many citizens took refuge in other territories | Choice 1  | 
# MAGIC |correct| The men fell unconscious | The assailant struckl the man in the head | he assailant s took the man's wallet. | choice1 | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for COPA
# MAGIC `.setTask('copa choice1:)`, prefix choice2 with `choice2:` , prefix premise with `premise:` and prefix the question with `question`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 COPA - Sentence Completion/ Binary choice selection
# MAGIC 
# MAGIC ```
# MAGIC copa 
# MAGIC choice1:   He fell off the ladder    
# MAGIC choice2:   He climbed up the lader       
# MAGIC premise:   The man lost his balance on the ladder 
# MAGIC question:  effect
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Task 11 [MultiRc - Question Answering](https://www.aclweb.org/anthology/N18-1023.pdf)
# MAGIC Evaluates an `answer` for a `question` as `true` or `false` based on an input `paragraph`
# MAGIC The T5 model predicts for a `question` and a `paragraph` of `sentences` wether an `answer` is true or not,
# MAGIC based on the semantic contents of the paragraph.        
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Exceeds human performance by a large margin**
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC | Question                                                     | Answer                                                              | Prediction | paragraph|
# MAGIC |--------------------------------------------------------------|---------------------------------------------------------------------|------------|----------|
# MAGIC | Why was Joey surprised the morning he woke up for breakfast? | There was only pie to eat, rather than traditional breakfast foods  |  True   |Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. He couldn’t find anything to eat except for pie! Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.,          |
# MAGIC | Why was Joey surprised the morning he woke up for breakfast? | There was a T-Rex in his garden  |  False   |Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. He couldn’t find anything to eat except for pie! Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.,          |
# MAGIC 
# MAGIC ## How to configure T5 task for MultiRC
# MAGIC `.setTask('multirc questions:)`  followed by `answer:` prefix for the answer to evaluate, followed by `paragraph:` and then a series of sentences, where each sentence is prefixed with `Sent n:`prefix second sentence with sentence2:
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 MultiRc task:
# MAGIC ```
# MAGIC multirc questions:  Why was Joey surprised the morning he woke up for breakfast?      
# MAGIC answer:             There was a T-REX in his garden.      
# MAGIC paragraph:      
# MAGIC Sent 1:             Once upon a time, there was a squirrel named Joey.      
# MAGIC Sent 2:             Joey loved to go outside and play with his cousin Jimmy.      
# MAGIC Sent 3:             Joey and Jimmy played silly games together, and were always laughing.      
# MAGIC Sent 4:             One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond.      
# MAGIC Sent 5:             Joey woke up early in the morning to eat some food before they left.      
# MAGIC Sent 6:             He couldn’t find anything to eat except for pie!      
# MAGIC Sent 7:             Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.      
# MAGIC Sent 8:             After he ate, he and Jimmy went to the pond.      
# MAGIC Sent 9:             On their way there they saw their friend Jack Rabbit.      
# MAGIC Sent 10:            They dove into the water and swam for several hours.      
# MAGIC Sent 11:            The sun was out, but the breeze was cold.      
# MAGIC Sent 12:            Joey and Jimmy got out of the water and started walking home.      
# MAGIC Sent 13:            Their fur was wet, and the breeze chilled them.      
# MAGIC Sent 14:            When they got home, they dried off, and Jimmy put on his favorite purple shirt.      
# MAGIC Sent 15:            Joey put on a blue shirt with red and green dots.      
# MAGIC Sent 16:            The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.      
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC # Task 12 [WiC - Word sense disambiguation](https://arxiv.org/abs/1808.09121)
# MAGIC Decide for `two sentence`s with a shared `disambigous word` wether they have the target word has the same `semantic meaning` in both sentences.       
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC |Predicted | disambigous word| Sentence 1 | Sentence 2 | 
# MAGIC |----------|-----------------|------------|------------|
# MAGIC | False    | kill            | He totally killed that rock show! | The airplane crash killed his family | 
# MAGIC | True     | window          | The expanded window will give us time to catch the thieves.|You have a two-hour window for turning in your homework. |     
# MAGIC | False     | window          | He jumped out of the window.|You have a two-hour window for turning in your homework. |     
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for MultiRC
# MAGIC `.setTask('wic pos:)`  followed by `sentence1:` prefix for the first sentence, followed by `sentence2:` prefix for the second sentence.
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5  WiC task:
# MAGIC 
# MAGIC ```
# MAGIC wic pos:
# MAGIC sentence1:    The expanded window will give us time to catch the thieves.
# MAGIC sentence2:    You have a two-hour window of turning in your homework.
# MAGIC word :        window
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Task 13 [WSC and DPR - Coreference resolution/ Pronoun ambiguity resolver  ](https://www.aaai.org/ocs/index.php/KR/KR12/paper/view/4492/0)
# MAGIC Predict for an `ambiguous pronoun` to which `noun` it is referring to.     
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC |Prediction| Text | 
# MAGIC |----------|-------|
# MAGIC | stable   | The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy. | 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for WSC/DPR
# MAGIC `.setTask('wsc:)` and surround pronoun with asteriks symbols..
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5  WSC/DPR  task:
# MAGIC The `ambiguous pronous` should be surrounded with `*` symbols.
# MAGIC 
# MAGIC ***Note*** Read [Appendix A.](https://arxiv.org/pdf/1910.10683.pdf#page=64&zoom=100,84,360) for more info
# MAGIC ```
# MAGIC wsc: 
# MAGIC The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy.
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC # Task 14 [Text summarization](https://arxiv.org/abs/1506.03340)
# MAGIC `Summarizes` a paragraph into a shorter version with the same semantic meaning.
# MAGIC 
# MAGIC | Predicted summary| Text | 
# MAGIC |------------------|-------|
# MAGIC | manchester united face newcastle in the premier league on wednesday . louis van gaal's side currently sit two points clear of liverpool in fourth . the belgian duo took to the dance floor on monday night with some friends .            | the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth . | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for summarization
# MAGIC `.setTask('summarize:)`
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 summarization task:
# MAGIC This task requires no pre-processing, setting the task to `summarize` is sufficient.
# MAGIC ```
# MAGIC the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .
# MAGIC ```
# MAGIC 
# MAGIC # Task 15 [SQuAD - Context based question answering](https://arxiv.org/abs/1606.05250)
# MAGIC Predict an `answer` to a `question` based on input `context`.
# MAGIC 
# MAGIC |Predicted Answer | Question | Context | 
# MAGIC |-----------------|----------|------|
# MAGIC |carbon monoxide| What does increased oxygen concentrations in the patient’s lungs displace? | Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
# MAGIC |pie| What did Joey eat for breakfast?| Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed,'|  
# MAGIC 
# MAGIC ## How to configure T5 task parameter for Squad Context based question answering
# MAGIC `.setTask('question:)` and prefix the context which can be made up of multiple sentences with `context:`
# MAGIC 
# MAGIC ## Example pre-processed input for T5 Squad Context based question answering:
# MAGIC ```
# MAGIC question: What does increased oxygen concentrations in the patient’s lungs displace? 
# MAGIC context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Task 16 [WMT1 Translate English to German](https://arxiv.org/abs/1706.03762)
# MAGIC For translation tasks use the `marian` model
# MAGIC ## How to configure T5 task parameter for WMT Translate English to German
# MAGIC `.setTask('translate English to German:)`
# MAGIC 
# MAGIC # Task 17 [WMT2 Translate English to French](https://arxiv.org/abs/1706.03762)
# MAGIC For translation tasks use the `marian` model
# MAGIC ## How to configure T5 task parameter for WMT Translate English to French
# MAGIC `.setTask('translate English to French:)`
# MAGIC 
# MAGIC 
# MAGIC # 18 [WMT3 - Translate English to Romanian](https://arxiv.org/abs/1706.03762)
# MAGIC For translation tasks use the `marian` model
# MAGIC ## How to configure T5 task parameter for English to Romanian
# MAGIC `.setTask('translate English to Romanian:)`

# COMMAND ----------

# MAGIC %md
# MAGIC # Spark-NLP Example for every Task:

# COMMAND ----------

import sparknlp

spark = sparknlp.start()

print("Spark NLP version", sparknlp.version())

print("Apache Spark version:", spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Document assembler and T5 model for running the tasks

# COMMAND ----------

import pandas as pd
pd.set_option('display.width', 100000)
pd.set_option('max_colwidth', 8000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# COMMAND ----------

from sparknlp.annotator import *
import sparknlp
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline


documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document") 

# Can take in document or sentence columns
t5 = T5Transformer.pretrained(name='t5_base',lang='en')\
  .setInputCols('document')\
  .setOutputCol("T5")\
  .setMaxOutputLength(200)


# COMMAND ----------

# MAGIC %md
# MAGIC # Task 1 [CoLA - Binary Grammatical Sentence acceptability classification](https://nyu-mll.github.io/CoLA/)
# MAGIC Judges if a sentence is grammatically acceptable.     
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Example
# MAGIC 
# MAGIC |sentence  | prediction|
# MAGIC |------------|------------|
# MAGIC | Anna and Mike is going skiing and they is liked is | unacceptable |      
# MAGIC | Anna and Mike like to dance | acceptable | 
# MAGIC 
# MAGIC ## How to configure T5 task for CoLA
# MAGIC `.setTask(cola sentence:)` prefix.
# MAGIC 
# MAGIC ### Example pre-processed input for T5 CoLA sentence acceptability judgement:
# MAGIC ```
# MAGIC cola 
# MAGIC sentence: Anna and Mike is going skiing and they is liked is
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('cola sentence:')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data
sentences = [['Anna and Mike is going skiing and they is liked is'],['Anna and Mike like to dance']]
df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 2 [RTE - Natural language inference Deduction Classification](https://dl.acm.org/doi/10.1007/11736790_9)
# MAGIC The RTE task is defined as recognizing, given two text fragments, whether the meaning of one text can be inferred (entailed) from the other or not.       
# MAGIC Classification of sentence pairs as entailment and not_entailment       
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and  [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Example
# MAGIC 
# MAGIC |sentence 1 | sentence 2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC Kessler ’s team conducted 60,643 interviews with adults in 14 countries.  |  Kessler ’s team interviewed more than 60,000 adults in 14 countries | entailment
# MAGIC Peter loves New York, it is his favorite city| Peter loves new York. | entailment
# MAGIC Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.  |Johnny is a millionare | entailment|
# MAGIC Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.  |Johnny is a poor man  | not_entailment | 
# MAGIC | It was raining in England for the last 4 weeks | England was very dry yesterday | not_entailment|
# MAGIC 
# MAGIC ## How to configure T5 task for RTE
# MAGIC `.setTask('rte sentence1:)` and prefix second sentence with `sentence2:`
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 RTE - 2 Class Natural language inference
# MAGIC ```
# MAGIC rte 
# MAGIC sentence1: Recent report say Peter makes he alot of money, he earned 10 million USD each year for the last 5 years. 
# MAGIC sentence2: Peter is a millionare.
# MAGIC ```
# MAGIC 
# MAGIC ### References
# MAGIC - https://arxiv.org/abs/2010.03061

# COMMAND ----------

# Set the task on T5
t5.setTask('rte sentence1:')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             ['Recent report say Peter makes he alot of money, he earned 10 million USD each year for the last 5 years.  sentence2: Peter is a millionare'],
             ['Recent report say Peter makes he alot of money, he earned 10 million USD each year for the last 5 years.  sentence2: Peter is a poor man']
            ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 3 [MNLI - 3 Class Natural Language Inference 3-class contradiction classification](https://arxiv.org/abs/1704.05426)
# MAGIC Classification of sentence pairs with the labels `entailment`, `contradiction`, and `neutral`.      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC This classifier predicts for two sentences :
# MAGIC - Whether the first sentence logically and semantically follows from the second sentence as entailment
# MAGIC - Whether the first sentence is a contradiction to the second sentence as a contradiction
# MAGIC - Whether the first sentence does not entail or contradict the first sentence as neutral
# MAGIC 
# MAGIC | Hypothesis | Premise | prediction|
# MAGIC |------------|------------|----------|
# MAGIC | Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years. |    Johnny is a poor man.  | contradiction|
# MAGIC |It rained in England the last 4 weeks.| It was snowing in New York last week| neutral | 
# MAGIC 
# MAGIC ## How to configure T5 task for MNLI
# MAGIC `.setTask('mnli hypothesis:)` and prefix second sentence with `premise:`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 MNLI - 3 Class Natural Language Inference
# MAGIC 
# MAGIC ```
# MAGIC mnli 
# MAGIC hypothesis: At 8:34, the Boston Center controller received a third, transmission from American 11.    
# MAGIC premise: The Boston Center controller got a third transmission from American 11.
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('mnli ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              ''' hypothesis: At 8:34, the Boston Center controller received a third, transmission from American 11.
                  premise: The Boston Center controller got a third transmission from American 11.
              '''
             ],
             ['''  
              hypothesis: Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.
              premise: Johnny is a poor man.
              ''']

             ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show()#.toPandas().head(5) <-- for better vis of result data frame

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 4 [MRPC - Binary Paraphrasing/ sentence similarity classification ](https://www.aclweb.org/anthology/I05-5002.pdf)
# MAGIC Detect whether one sentence is a re-phrasing or similar to another sentence      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC | Sentence1 | Sentence2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .| Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 " . | equivalent | 
# MAGIC | I like to eat peanutbutter for breakfast| I like to play football | not_equivalent | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for MRPC
# MAGIC `.setTask('mrpc sentence1:)` and prefix second sentence with `sentence2:`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 MRPC - Binary Paraphrasing/ sentence similarity
# MAGIC 
# MAGIC ```
# MAGIC mrpc 
# MAGIC sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said . 
# MAGIC sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11",
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('mrpc ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              ''' sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .
                  sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 " 
              '''
             ],
             ['''  
              sentence1: I like to eat peanutbutter for breakfast
              sentence2: 	I like to play football.
              ''']
             ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).toPandas()#show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 5 [QNLI - Natural Language Inference question answered classification](https://arxiv.org/pdf/1804.07461.pdf)
# MAGIC Classify whether a question is answered by a sentence (`entailed`).       
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC | Question | Answer | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |Where did Jebe die?| Ghenkis Khan recalled Subtai back to Mongolia soon afterward, and Jebe died on the road back to Samarkand | entailment|
# MAGIC |What does Steve like to eat? | Steve watches TV all day | not_netailment
# MAGIC 
# MAGIC ## How to configure T5 task for QNLI - Natural Language Inference question answered classification
# MAGIC `.setTask('QNLI sentence1:)` and prefix question with `question:` sentence with `sentence:`:
# MAGIC 
# MAGIC ### Example pre-processed input for T5 QNLI - Natural Language Inference question answered classification
# MAGIC 
# MAGIC ```
# MAGIC qnli
# MAGIC question: Where did Jebe die?     
# MAGIC sentence: Ghenkis Khan recalled Subtai back to Mongolia soon afterwards, and Jebe died on the road back to Samarkand,
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('QNLI ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              ''' question:  Where did Jebe die?    
                  sentence: Ghenkis Khan recalled Subtai back to Mongolia soon afterwards, and Jebe died on the road back to Samarkand,
              '''
             ],
             ['''  
              question: What does Steve like to eat?	
              sentence: 	Steve watches TV all day
              ''']

             ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).toPandas()#.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 6 [QQP - Binary Question Similarity/Paraphrasing](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
# MAGIC Based on a quora dataset, determine whether a pair of questions are semantically equivalent.      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC | Question1 | Question2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |What attributes would have made you highly desirable in ancient Rome?        | How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER? | not_duplicate | 
# MAGIC |What was it like in Ancient rome?  | What was Ancient rome like?| duplicate | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for QQP
# MAGIC .setTask('qqp question1:) and
# MAGIC prefix second sentence with question2:
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 QQP - Binary Question Similarity/Paraphrasing
# MAGIC 
# MAGIC ```
# MAGIC qqp 
# MAGIC question1: What attributes would have made you highly desirable in ancient Rome?        
# MAGIC question2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('qqp ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              ''' question1:  What attributes would have made you highly desirable in ancient Rome?    
                  question2:  How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?'
              '''
             ],
             ['''  
              question1: What was it like in Ancient rome?
              question2: 	What was Ancient rome like?
              ''']

             ]


df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).toPandas()#.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 7 [SST2 - Binary Sentiment Analysis](https://www.aclweb.org/anthology/D13-1170.pdf)
# MAGIC Binary sentiment classification.      
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC | Sentence1 | Prediction  | 
# MAGIC |-----------|-----------|
# MAGIC |it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight |  positive| 
# MAGIC |I really hated that movie | negative | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for  SST2
# MAGIC `.setTask('sst2 sentence: ')`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 SST2 - Binary Sentiment Analysis
# MAGIC 
# MAGIC ```
# MAGIC sst2
# MAGIC sentence: I hated that movie
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('sst2 sentence: ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              ''' I really hated that movie'''],
             ['''  it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight''']]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).toPandas()#show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Task8 [STSB - Regressive semantic sentence similarity](https://www.aclweb.org/anthology/S17-2001/)
# MAGIC Measures how similar two sentences are on a scale from 0 to 5 with 21 classes representing a regressive label.     
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf).
# MAGIC 
# MAGIC 
# MAGIC | Question1 | Question2 | prediction|
# MAGIC |------------|------------|----------|
# MAGIC |What attributes would have made you highly desirable in ancient Rome?        | How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER? | 0 | 
# MAGIC |What was it like in Ancient rome?  | What was Ancient rome like?| 5.0 | 
# MAGIC |What was live like as a King in Ancient Rome??       | What is it like to live in Rome? | 3.2 | 
# MAGIC 
# MAGIC ## How to configure T5 task for STSB
# MAGIC `.setTask('stsb sentence1:)` and prefix second sentence with `sentence2:`
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 STSB - Regressive semantic sentence similarity
# MAGIC 
# MAGIC ```
# MAGIC stsb
# MAGIC sentence1: What attributes would have made you highly desirable in ancient Rome?        
# MAGIC sentence2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('stsb ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              ''' sentence1:  What attributes would have made you highly desirable in ancient Rome?  
                  sentence2:  How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?'
              '''
             ],
             ['''  
              sentence1: What was it like in Ancient rome?
              sentence2: 	What was Ancient rome like?
              '''],
              ['''  
              sentence1: What was live like as a King in Ancient Rome??
              sentence2: 	What was Ancient rome like?
              ''']
             ]

df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).toPandas()#show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 9 [CB -  Natural language inference contradiction classification](https://ojs.ub.uni-konstanz.de/sub/index.php/sub/article/view/601)
# MAGIC Classify whether a Premise contradicts a Hypothesis.    
# MAGIC Predicts entailment, neutral and contradiction     
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC | Hypothesis | Premise | Prediction | 
# MAGIC |--------|-------------|----------|
# MAGIC |Valence was helping | Valence the void-brain, Valence the virtuous valet. Why couldn’t the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping'| Contradiction|
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for CB
# MAGIC `.setTask('cb hypothesis:)` and prefix premise with `premise:`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 CB -  Natural language inference contradiction classification
# MAGIC 
# MAGIC ```
# MAGIC cb 
# MAGIC hypothesis: Valence was helping      
# MAGIC premise: Valence the void-brain, Valence the virtuous valet. Why couldn’t the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping,
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('cb ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              '''
              hypothesis: Recent report say Johnny makes he alot of money, he earned 10 million USD each year for the last 5 years.
              premise: Johnny is a poor man.
                            ''']
             ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).toPandas()#show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 10 [COPA - Sentence Completion/ Binary choice selection](https://www.aaai.org/ocs/index.php/SSS/SSS11/paper/view/2418/0)
# MAGIC The Choice of Plausible Alternatives (COPA) task by Roemmele et al. (2011) evaluates
# MAGIC causal reasoning between events, which requires commonsense knowledge about what usually takes
# MAGIC place in the world. Each example provides a premise and either asks for the correct cause or effect
# MAGIC from two choices, thus testing either ``backward`` or `forward causal reasoning`. COPA data, which
# MAGIC consists of 1,000 examples total, can be downloaded at https://people.ict.usc.e
# MAGIC 
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC This classifier selects from a choice of `2 options` which one the correct is based on a `premise`.
# MAGIC 
# MAGIC 
# MAGIC ## forward causal reasoning
# MAGIC Premise: The man lost his balance on the ladder.     
# MAGIC question: What happened as a result?        
# MAGIC Alternative 1: He fell off the ladder.       
# MAGIC Alternative 2: He climbed up the ladder.
# MAGIC ## backwards causal reasoning
# MAGIC Premise: The man fell unconscious. What was the cause
# MAGIC of this?       
# MAGIC Alternative 1: The assailant struck the man in the head.      
# MAGIC Alternative 2: The assailant took the man’s wallet.
# MAGIC 
# MAGIC 
# MAGIC | Question | Premise | Choice 1 | Choice  2 | Prediction | 
# MAGIC |--------|-------------|----------|---------|-------------|
# MAGIC |effect | Politcal Violence broke out in the nation. | many citizens relocated to the capitol. |  Many citizens took refuge in other territories | Choice 1  | 
# MAGIC |correct| The men fell unconscious | The assailant struckl the man in the head | he assailant s took the man's wallet. | choice1 | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for COPA
# MAGIC `.setTask('copa choice1:)`, prefix choice2 with `choice2:` , prefix premise with `premise:` and prefix the question with `question`
# MAGIC 
# MAGIC ### Example pre-processed input for T5 COPA - Sentence Completion/ Binary choice selection
# MAGIC 
# MAGIC ```
# MAGIC copa 
# MAGIC choice1:   He fell off the ladder    
# MAGIC choice2:   He climbed up the lader       
# MAGIC premise:   The man lost his balance on the ladder 
# MAGIC question:  effect
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('copa ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              '''
            choice1:   He fell off the ladder    
            choice2:   He climbed up the lader       
            premise:   The man lost his balance on the ladder 
            question:  effect

                            ''']
             ]


df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).toPandas()#show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 11 [MultiRc - Question Answering](https://www.aclweb.org/anthology/N18-1023.pdf)
# MAGIC Evaluates an `answer` for a `question` as `true` or `false` based on an input `paragraph`
# MAGIC The T5 model predicts for a `question` and a `paragraph` of `sentences` wether an `answer` is true or not,
# MAGIC based on the semantic contents of the paragraph.        
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Exceeds human performance by a large margin**
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC | Question                                                     | Answer                                                              | Prediction | paragraph|
# MAGIC |--------------------------------------------------------------|---------------------------------------------------------------------|------------|----------|
# MAGIC | Why was Joey surprised the morning he woke up for breakfast? | There was only pie to eat, rather than traditional breakfast foods  |  True   |Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. He couldn’t find anything to eat except for pie! Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.,          |
# MAGIC | Why was Joey surprised the morning he woke up for breakfast? | There was a T-Rex in his garden  |  False   |Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. He couldn’t find anything to eat except for pie! Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.,          |
# MAGIC 
# MAGIC ## How to configure T5 task for MultiRC
# MAGIC `.setTask('multirc questions:)`  followed by `answer:` prefix for the answer to evaluate, followed by `paragraph:` and then a series of sentences, where each sentence is prefixed with `Sent n:`prefix second sentence with sentence2:
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 MultiRc task:
# MAGIC ```
# MAGIC multirc questions:  Why was Joey surprised the morning he woke up for breakfast?      
# MAGIC answer:             There was a T-REX in his garden.      
# MAGIC paragraph:      
# MAGIC Sent 1:             Once upon a time, there was a squirrel named Joey.      
# MAGIC Sent 2:             Joey loved to go outside and play with his cousin Jimmy.      
# MAGIC Sent 3:             Joey and Jimmy played silly games together, and were always laughing.      
# MAGIC Sent 4:             One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond.      
# MAGIC Sent 5:             Joey woke up early in the morning to eat some food before they left.      
# MAGIC Sent 6:             He couldn’t find anything to eat except for pie!      
# MAGIC Sent 7:             Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.      
# MAGIC Sent 8:             After he ate, he and Jimmy went to the pond.      
# MAGIC Sent 9:             On their way there they saw their friend Jack Rabbit.      
# MAGIC Sent 10:            They dove into the water and swam for several hours.      
# MAGIC Sent 11:            The sun was out, but the breeze was cold.      
# MAGIC Sent 12:            Joey and Jimmy got out of the water and started walking home.      
# MAGIC Sent 13:            Their fur was wet, and the breeze chilled them.      
# MAGIC Sent 14:            When they got home, they dried off, and Jimmy put on his favorite purple shirt.      
# MAGIC Sent 15:            Joey put on a blue shirt with red and green dots.      
# MAGIC Sent 16:            The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.      
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('multirc ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              '''
questions:  Why was Joey surprised the morning he woke up for breakfast?      
answer:             There was a T-REX in his garden.      
paragraph:      
Sent 1:             Once upon a time, there was a squirrel named Joey.      
Sent 2:             Joey loved to go outside and play with his cousin Jimmy.      
Sent 3:             Joey and Jimmy played silly games together, and were always laughing.      
Sent 4:             One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond.      
Sent 5:             Joey woke up early in the morning to eat some food before they left.      
Sent 6:             He couldn’t find anything to eat except for pie!      
Sent 7:             Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.      
Sent 8:             After he ate, he and Jimmy went to the pond.      
Sent 9:             On their way there they saw their friend Jack Rabbit.      
Sent 10:            They dove into the water and swam for several hours.      
Sent 11:            The sun was out, but the breeze was cold.      
Sent 12:            Joey and Jimmy got out of the water and started walking home.      
Sent 13:            Their fur was wet, and the breeze chilled them.      
Sent 14:            When they got home, they dried off, and Jimmy put on his favorite purple shirt.      
Sent 15:            Joey put on a blue shirt with red and green dots.      
Sent 16:            The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.      

                            '''],
             
             [
              '''
questions:  Why was Joey surprised the morning he woke up for breakfast?      
answer:             There was only pie for breakfast.      
paragraph:      
Sent 1:             Once upon a time, there was a squirrel named Joey.      
Sent 2:             Joey loved to go outside and play with his cousin Jimmy.      
Sent 3:             Joey and Jimmy played silly games together, and were always laughing.      
Sent 4:             One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond.      
Sent 5:             Joey woke up early in the morning to eat some food before they left.      
Sent 6:             He couldn’t find anything to eat except for pie!      
Sent 7:             Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.      
Sent 8:             After he ate, he and Jimmy went to the pond.      
Sent 9:             On their way there they saw their friend Jack Rabbit.      
Sent 10:            They dove into the water and swam for several hours.      
Sent 11:            The sun was out, but the breeze was cold.      
Sent 12:            Joey and Jimmy got out of the water and started walking home.      
Sent 13:            Their fur was wet, and the breeze chilled them.      
Sent 14:            When they got home, they dried off, and Jimmy put on his favorite purple shirt.      
Sent 15:            Joey put on a blue shirt with red and green dots.      
Sent 16:            The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.      

                            ''']
                  
             
             ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 12 [WiC - Word sense disambiguation](https://arxiv.org/abs/1808.09121)
# MAGIC Decide for `two sentence`s with a shared `disambigous word` wether they have the target word has the same `semantic meaning` in both sentences.       
# MAGIC This is a sub-task of [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC 
# MAGIC |Predicted | disambigous word| Sentence 1 | Sentence 2 | 
# MAGIC |----------|-----------------|------------|------------|
# MAGIC | False    | kill            | He totally killed that rock show! | The airplane crash killed his family | 
# MAGIC | True     | window          | The expanded window will give us time to catch the thieves.|You have a two-hour window for turning in your homework. |     
# MAGIC | False     | window          | He jumped out of the window.|You have a two-hour window for turning in your homework. |     
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for MultiRC
# MAGIC `.setTask('wic pos:)`  followed by `sentence1:` prefix for the first sentence, followed by `sentence2:` prefix for the second sentence.
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5  WiC task:
# MAGIC 
# MAGIC ```
# MAGIC wic pos:
# MAGIC sentence1:    The expanded window will give us time to catch the thieves.
# MAGIC sentence2:    You have a two-hour window of turning in your homework.
# MAGIC word :        window
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('wic ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              '''
pos:
sentence1:    The expanded window will give us time to catch the thieves.
sentence2:    You have a two-hour window of turning in your homework.
word :        window

                            '''],]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=180)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 13 [WSC and DPR - Coreference resolution/ Pronoun ambiguity resolver  ](https://www.aaai.org/ocs/index.php/KR/KR12/paper/view/4492/0)
# MAGIC Predict for an `ambiguous pronoun` to which `noun` it is referring to.     
# MAGIC This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and [SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf).
# MAGIC 
# MAGIC |Prediction| Text | 
# MAGIC |----------|-------|
# MAGIC | stable   | The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy. | 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for WSC/DPR
# MAGIC `.setTask('wsc:)` and surround pronoun with asteriks symbols..
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5  WSC/DPR  task:
# MAGIC The `ambiguous pronous` should be surrounded with `*` symbols.
# MAGIC 
# MAGIC ***Note*** Read [Appendix A.](https://arxiv.org/pdf/1910.10683.pdf#page=64&zoom=100,84,360) for more info
# MAGIC ```
# MAGIC wsc: 
# MAGIC The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy.
# MAGIC ```

# COMMAND ----------

# Does not work yet 100% correct
# Set the task on T5
t5.setTask('wsc')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [['''The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy.'''],]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 14 [Text summarization](https://arxiv.org/abs/1506.03340)
# MAGIC `Summarizes` a paragraph into a shorter version with the same semantic meaning.
# MAGIC 
# MAGIC | Predicted summary| Text | 
# MAGIC |------------------|-------|
# MAGIC | manchester united face newcastle in the premier league on wednesday . louis van gaal's side currently sit two points clear of liverpool in fourth . the belgian duo took to the dance floor on monday night with some friends .            | the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth . | 
# MAGIC 
# MAGIC 
# MAGIC ## How to configure T5 task for summarization
# MAGIC `.setTask('summarize:)`
# MAGIC 
# MAGIC 
# MAGIC ### Example pre-processed input for T5 summarization task:
# MAGIC This task requires no pre-processing, setting the task to `summarize` is sufficient.
# MAGIC ```
# MAGIC the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('summarize ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              '''
The belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .
                            '''],
             ['''  Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations. It has two major branches, differential calculus and integral calculus; the former concerns instantaneous rates of change, and the slopes of curves, while integral calculus concerns accumulation of quantities, and areas under or between curves. These two branches are related to each other by the fundamental theorem of calculus, and they make use of the fundamental notions of convergence of infinite sequences and infinite series to a well-defined limit.[1] Infinitesimal calculus was developed independently in the late 17th century by Isaac Newton and Gottfried Wilhelm Leibniz.[2][3] Today, calculus has widespread uses in science, engineering, and economics.[4] In mathematics education, calculus denotes courses of elementary mathematical analysis, which are mainly devoted to the study of functions and limits. The word calculus (plural calculi) is a Latin word, meaning originally "small pebble" (this meaning is kept in medicine – see Calculus (medicine)). Because such pebbles were used for calculation, the meaning of the word has evolved and today usually means a method of computation. It is therefore used for naming specific methods of calculation and related theories, such as propositional calculus, Ricci calculus, calculus of variations, lambda calculus, and process calculus.''']
             ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 15 [SQuAD - Context based question answering](https://arxiv.org/abs/1606.05250)
# MAGIC Predict an `answer` to a `question` based on input `context`.
# MAGIC 
# MAGIC |Predicted Answer | Question | Context | 
# MAGIC |-----------------|----------|------|
# MAGIC |carbon monoxide| What does increased oxygen concentrations in the patient’s lungs displace? | Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
# MAGIC |pie| What did Joey eat for breakfast?| Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed,'|  
# MAGIC 
# MAGIC ## How to configure T5 task parameter for Squad Context based question answering
# MAGIC `.setTask('question:)` and prefix the context which can be made up of multiple sentences with `context:`
# MAGIC 
# MAGIC ## Example pre-processed input for T5 Squad Context based question answering:
# MAGIC ```
# MAGIC question: What does increased oxygen concentrations in the patient’s lungs displace? 
# MAGIC context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
# MAGIC ```

# COMMAND ----------

# Set the task on T5
t5.setTask('question: ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              '''
What does increased oxygen concentrations in the patient’s lungs displace? 
context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
                            ''']
             ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 16 [WMT1 Translate English to German](https://arxiv.org/abs/1706.03762)
# MAGIC For translation tasks use the `marian` model
# MAGIC ## How to configure T5 task parameter for WMT Translate English to German
# MAGIC `.setTask('translate English to German:)`

# COMMAND ----------

# Set the task on T5
t5.setTask('translate English to German: ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [
              '''I like sausage and Tea for breakfast with potatoes'''],]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 17 [WMT2 Translate English to French](https://arxiv.org/abs/1706.03762)
# MAGIC For translation tasks use the `marian` model
# MAGIC ## How to configure T5 task parameter for WMT Translate English to French
# MAGIC `.setTask('translate English to French:)`

# COMMAND ----------

# Set the task on T5
t5.setTask('translate English to French: ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             [ '''I like sausage and Tea for breakfast with potatoes''']
            ]



df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 18 [WMT3 - Translate English to Romanian](https://arxiv.org/abs/1706.03762)
# MAGIC For translation tasks use the `marian` model
# MAGIC ## How to configure T5 task parameter for English to Romanian
# MAGIC `.setTask('translate English to Romanian:)`

# COMMAND ----------

# Set the task on T5
t5.setTask('translate English to Romanian: ')

# Build pipeline with T5
pipe_components = [documentAssembler,t5]
pipeline = Pipeline().setStages( pipe_components)

# define Data, add additional tags between sentences
sentences = [
             ['''I like sausage and Tea for breakfast with potatoes''']
            ]

df = spark.createDataFrame(sentences).toDF("text")

#Predict on text data with T5
model = pipeline.fit(df)
annotated_df = model.transform(df)
annotated_df.select(['text','t5.result']).show(truncate=False)

# COMMAND ----------

