import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from pyspark.context import SparkContext,SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import CountVectorizer,Tokenizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import size
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf, col, explode, arrays_zip

# extract vocabulary from CountVectorizer
def extract_vocabulary(cv_model,model):
    vocab = cv_model.vocabulary
    topics = model.describeTopics()
    topics_rdd = topics.rdd
    topics_words = topics_rdd\
           .map(lambda row: row['termIndices'])\
           .map(lambda idx_list: [vocab[idx] for idx in idx_list])\
           .collect()
    return topics_words

def termsIdx2Term(vocabulary):
    def termsIdx2Term(termIndices):
        return [vocabulary[int(index)] for index in termIndices]
    return udf(termsIdx2Term, ArrayType(StringType()))

#Build the LDA Model
def model_topic(vectorized_tokens,start_topic_count=10, end_topic_count=100, step=10):
    perplexity_list = []
    model_list = []
    for num_topics in tqdm(range(start_topic_count, end_topic_count + 1, step)):
        lda = LDA(k=num_topics, maxIter=30)
        model = lda.fit(vectorized_tokens)
        lp = model.logPerplexity(vectorized_tokens)
        model_list.append(model)
        perplexity_list.append(lp)
    return model_list,perplexity_list

if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.driver.memory","15g")
    conf.set("spark.executor.memory", "10g")
    conf.set("spark.memory.offHeap.enabled", "True")
    conf.set("spark.memory.offHeap.size", "2g")
    conf.set("spark.driver.maxResultSize", "20g")
    conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    data = spark.read.option("header", True).csv("topic_modeling_inputs_noempty_no_token.csv")
    print((data.count(), len(data.columns)))
    # tokens
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokensData = tokenizer.transform(data)
    # feature enginerring
    cv = CountVectorizer(inputCol="words", vocabSize=3000, outputCol="features", minDF=5)
    cv_model = cv.fit(tokensData)
    vectorized_tokens = cv_model.transform(tokensData)

    #find the best number of topics for model
    model_list,perplexity_list,topics_list,stopword,topics_words_list= model_topic(tokensData，start_topic_count=10, end_topic_count=40, step=10)
    score_df = pd.DataFrame({'#Topics': range(10, 30, 10), 'perplexity': np.round(perplexity_list, 4)})

    #draw the perplexity plot(for select number of topic)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(16, 6))
    sns.lineplot(x='#Topics', y='perplexity', data=score_df)
    plt.show()
    num_topics = score_df['#Topics'][0]

    # final model
    num_topics = 30
    lda = LDA(k=num_topics, maxIter=100)
    final_model = lda.fit(vectorized_tokens)
    lp = final_model.logPerplexity(vectorized_tokens)
    print(lp)

    # Get topic words and weight and save to CSV
    topics = final_model.describeTopics()
    topics_pandas = topics.toPandas()
    topics_words = extract_vocabulary(cv_model,final_model)
    topics_pandas.insert(topics_pandas.shape[1], 'Topic_word', topics_words)
    topics_pandas.to_csv('topics30.csv')
    topics_pandas.show()

    # show topics
    vocabList = cv_model.vocabulary
    final = final_model.describeTopics().withColumn("terms", termsIdx2Term(vocabList)("termIndices"))
    final = final.withColumn("no_of_terms", size("terms"))
    topicsTmp = final.withColumn("termWithProb", explode(arrays_zip(col("terms"), col("termWeights")))).select(
        "topic",
        "termWithProb.terms", "termWithProb.termWeights")
    topicsTmp = topicsTmp.toPandas()
    topicsTmp.to_csv('topics30_word.csv')

    # Shows the result
    result = final_model.transform(vectorized_tokens)
    result = result.select('tic','datadate','words','features','topicDistribution')
    result_pandas = result.toPandas()
    result_pandas.to_csv('result_30.csv')

    #test model
    from pyspark.ml.clustering import LocalLDAModel
    final_model.save("LDA_model_30_saved")
    newModel = LocalLDAModel.load("LDA_model_30_saved")
    test = spark.read.option("header", True).csv("nona-test.csv")
    #tokenize
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokens_test = tokenizer.transform(test)
    tokens_test = tokens_test.select('datadate', 'tic', 'words')
    tokens_test = tokens_test.withColumn("no_of_tokens", size("words"))
    # Apply CountVectorizer
    vectorized_tokens_test = cv_model.transform(tokens_test)
    vectorized_tokens_test.show()

    transformed_test = newModel.transform(vectorized_tokens_test)
    transformed_test = transformed_test.select('datadate','tic','topicDistribution')
    transformed_test.show()
    transformed_test_pandas = transformed_test.toPandas()
    transformed_test_pandas.to_csv('topic_30_test.csv')

































