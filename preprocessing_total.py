"""
---README---
This code will run with all of your CPUs!!! (changeable, see row 23).
You will see a progress monitor in the output box like this
    xx%| finished_iterations/total_iterations [already_run_time<estimated_remain_time, xx it/s]
input: train.sql & test.sql (in Google drive, import to mysql)
output: mysql table with posts updated to processed texts
**********************************************************************************************
preprocessing details:
html tag, accented char, special char -> remove
Username and ticker -> remove all words that start with the @ symbol, keep tickers
URLs ->  remove
Repeated letters -> any letter occurring more than two times in a row is replaced with two occurrences
Retweets -> keep those with “RT”
Emoticons -> remove
digit -> remove
stopwords -> remove
ticker in comment -> remove
single character -> remove
special word like pic, th -> remove
all -> lowercase, lemmatization
**********************************************************************************************
Please change the followings before running this code:
1. connection details (see row 109 & 126)
2. [i, j] (see row 108)
3. if only want to run with n CPUs, change 'with Pool() as p' to 'with Pool(n) as p' (see row 111)
   to see how many CPUs your laptop has:
    import multiprocessing
    multiprocessing.cpu_count()
"""

import pymysql
import re
import pandas as pd
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from multiprocessing import Pool
from tqdm import tqdm
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # NFKD or Normalization Form Compatibility Decomposition: Characters are decomposed by compatibility,
    # and multiple combining characters are arranged in a specific order.
    return text


def remove_special_characters(text):
    pattern = r'[^a-zA-Z\s]' # remove digits
    text = re.sub(pattern, '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)  # correct repeat letters
    return text


def remove_stopwords(text):
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_noise(text):
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    cleaned_tokens = list()
    for token in tokens:
        cleaned_token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                               '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        cleaned_token = re.sub("(@[A-Za-z0-9_]+)", "", cleaned_token)
        cleaned_tokens.append(cleaned_token)
    cleaned_tokens = ' '.join(cleaned_tokens)
    return cleaned_tokens


def lemmatize_text(text, nlp):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def normalize_corpus(doc, nlp):
    # doc = re.sub("(\$[A-Z]+)",'', doc) # remove tickers
    doc = remove_accented_chars(doc)
    doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
    doc = remove_noise(doc)
    doc = doc.lower()
    doc = lemmatize_text(doc, nlp)
    special_char_pattern = re.compile(r'([{.(-)!}])')
    doc = special_char_pattern.sub(" \\1 ", doc)
    doc = remove_special_characters(doc)
    doc = re.sub(' +', ' ', doc)
    doc = remove_stopwords(doc)
    return doc


def keep_common(text, common):
    words = text.split()
    text = [word for word in words if word in common]
    text = ' '.join(text)
    return text


if __name__ == "__main__":
    i, j = 4000000, 5000000
    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select* from train " + "limit " + str(i) + "," + str(j - i)
    data = pd.read_sql(sqlcmd, conn)
    conn.close()
    raw_text = data['text']
    import functools
    import spacy
    nlp = spacy.load('en_core_web_sm')
    func = functools.partial(normalize_corpus, nlp=nlp)
    with Pool() as p:
        pro_text = list(tqdm(p.imap(func, raw_text, chunksize=2000), total=raw_text.shape[0]))
    p.close()
    p.join()
    data['text'] = pro_text

    # remove common word
    common_words = list()
    with open('google-10000-english.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            common_words.append(line.strip())
    print('Successfully read the common words!')
    raw_text = data['text']
    import functools
    from multiprocessing import Pool
    import time
    func = functools.partial(keep_common, common=common_words)
    with Pool() as p:
        pro_text = list(tqdm(p.imap(func, raw_text, chunksize=2000), total=raw_text.shape[0]))
    p.close()
    p.join()
    print('Successfully process the data!')
    start = time.time()
    text_lst = []

    # remove single character and special word
    special_lst = list('qwertyuiopasdfghjklzxcvbnm')
    special_lst.extend(['pic', 'html', 'inc', 'th', 'ya'])
    for i in range(data.shape[0]):
        text = pro_text[i].split()
        try:
            text.remove(data.ticker[i].replace('$', '').lower())
            for sp in special_lst:
                try:
                    text.remove(sp)
                except:
                    continue
            text = ' '.join(text)
            text_lst.append(text)
        except:
            text = ' '.join(text)
            text_lst.append(text)
        if i % 1000000 == 0:
            end = time.time()
            print((end - start) / 60, 'min, 1 million done')
    data['text'] = text_lst
    conn = create_engine('mysql+pymysql://root:******@localhost/Twitter', encoding='utf8')
    pd.io.sql.to_sql(data, "preprocessed_text_%s_%s" %(i, j), conn, if_exists='replace')
    print('All done!')