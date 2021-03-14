import pandas as pd
import numpy as np
import re
import nltk
from tqdm import tqdm
import gensim
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath

def extract_topic(topic_vector,id):
    for x,y in topic_vector:
        if x==id:
            return y
    return 0

def normalize(articles):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    token = articles.apply(tokenizer.tokenize)
    return token

data_all = pd.read_csv('TM_inputs_wotoken.csv')
print('Total number of documents:',data_all.shape[0])

data_all['words'] = normalize(data_all['text'])

texts = data_all['words']
dictionary = Dictionary(texts)
#feature engineering
corpus = [dictionary.doc2bow(text) for text in texts]
print('Total number of documents:', len(corpus))

lda = LdaMulticore(corpus, id2word=dictionary, num_topics=30)

#perplexity = lda.log_perplexity(corpus)
#print('Model Perplexity:', np.round(perplexity, 4))
data_all['topic_vector']=lda[corpus]

for x in range(30):
    data_all["t" + str(x + 1)] =data_all.apply(lambda r: extract_topic(r['topic_vector'],x),axis=1)

data_all.to_csv('twitter_lda_train.csv')

# Save model
temp_file = datapath("model")
lda.save(temp_file)

# Load a potentially pretrained model from disk.
lda_model = lda.load(temp_file)

# topic list with top 50 words
topics = [[(term, round(wt, 4)) for term, wt in lda_model.show_topic(n, topn=20)]
          for n in range(0, 30)]
#pd.set_option('display.max_colwidth', -1)
#topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics],
                         #columns=['Terms per Topic'],
                        # index=['Topic ' + str(t) for t in range(1, 30 + 1)])

l = pd.DataFrame(columns=['Topic','Term','Weight'])
topic_before = ['Topic ' + str(t) for t in range(1, 30 + 1)]
topic_after = [val for val in index_before for i in range(20)]
l.Topic = topic_after
term_list = []
weight_list = []
for topic in topics:
    term_list.extend([term for term, wt in topic])
    weight_list.extend([wt for term, wt in topic])
l.Term = term_list
l.Weight = weight_list

l.to_csv('T30topic_term&weight.csv')

# appled the lda model to test data
data_test = pd.read_csv('posts_test.csv')
data_test = data_test.rename(columns={'days_posts':'text'})
data_test['words'] = normalize(data_test['text'])
test_texts = data_test['words']
test_dictionary = Dictionary(test_texts)
other_corpus = [test_dictionary.doc2bow(text) for text in test_texts]
unseen_doc = other_corpus[index]
vector = lda[unseen_doc]
for x in range(30):
    data_test["t" + str(x + 1)] = data_test.apply(lambda r: extract_topic(vector, x), axis=1)
data_test.to_csv('twitter_lda_test.csv')