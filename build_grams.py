# -*- coding: utf-8 -*-
# author：Kyle time:6/3/2021
import gc
from collections import Counter
from multiprocessing import Pool
import pandas as pd
import pymysql
from tqdm import tqdm


def to_grams(args):  # return a list
    ticker, doc = args
    words = doc.split()
    ticker = str.lower(ticker)[1:]
    bi_list = list()
    while 1:
        try:
            words.remove(ticker)
        except:
            break
    for i in range(len(words) - 1):
        bi_list.append(words[i] + ' ' + words[i + 1])
    return [words, bi_list]


if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select ticker, text from clean2_train"
    data = pd.read_sql(sqlcmd, conn)
    conn.close()
    print('Successfully read the sql data!')
    args = zip(data['ticker'], data['text'])
    with Pool() as p:
        grams = list(tqdm(p.imap(to_grams, args, chunksize=5000), total=data.shape[0]))
    p.close()
    p.join()
    del data
    print(gc.collect())
    print('Successfully process the text!')
    unigrams = list()
    bigrams = list()
    for gram in tqdm(grams):
        unigrams += gram[0]
        bigrams += gram[1]
    del grams
    print(gc.collect())
    uni_temp = Counter(unigrams)
    bi_temp = Counter(bigrams)
    n = 2000
    uni_top = uni_temp.most_common(n)
    bi_top = bi_temp.most_common(n)
    del unigrams, bigrams
    print(gc.collect())
    print('Successfully extract the top unigrams and bigrams!')
    df1 = pd.DataFrame(columns=['unigram', 'num'])
    df2 = pd.DataFrame(columns=['bigram', 'num'])
    for i in range(n):
        df1.loc[i] = uni_top[i]
        df2.loc[i] = bi_top[i]
    df1.to_csv('top_%s_unigrams.csv' % n)
    df2.to_csv('top_%s_bigrams.csv' % n)
    print('Successfully save the result!')
    print('Congratulations! All done!')
