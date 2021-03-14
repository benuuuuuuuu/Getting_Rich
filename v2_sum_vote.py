# -*- coding: utf-8 -*-
# author：Kyle time:7/3/2021
import gc
from multiprocessing import Manager, Pool

import pymysql
import pandas as pd
import numpy as np
from tqdm import tqdm


def build_grams(args, grams):
    text, idx = args
    words = text.split()
    l = len(words)
    id_list = list()
    for i in range(l):
        if words[i] in grams:
            id_list.append((grams.index(words[i]), idx))
            if i < (l-1):
                bi = words[i] + ' ' + words[i+1]
                if bi in grams:
                    id_list.append((grams.index(bi), idx))
    return id_list


if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select ticker,created_time,text from clean2_train"
    sqlcmd2 = "select p1,p2,p3,p4,p5,p6 from price_concat"
    data = pd.read_sql(sqlcmd, conn)
    price = pd.read_sql(sqlcmd2, conn)
    conn.close()
    print('Successfully read the sql data!')

    data.columns = ['ticker', 'date', 'text']
    ticker_date = data[['ticker', 'date']].copy()
    ticker_date.drop_duplicates(keep='first', inplace=True, subset=['ticker', 'date'], ignore_index=True)
    ticker_date.reset_index(inplace=True)
    data = data.merge(ticker_date, how='left', on=['ticker', 'date'])
    data.drop(['ticker', 'date'], axis=1, inplace=True)
    data.columns = ['text', 'idx']
    print('Successfully merge the text and index!')

    del ticker_date
    print(gc.collect())

    bigrams = pd.read_csv('top_2000_bigrams.csv', index_col=0)
    bigrams.columns = ['grams', 'num']
    unigrams = pd.read_csv('top_2000_unigrams.csv', index_col=0)
    unigrams.columns = ['grams', 'num']
    top_grams = pd.concat([unigrams, bigrams], ignore_index=True)
    #  df = pd.DataFrame(columns = ['grams', '<-0.1', '-0.1_0.01', '-0.01_0', '0_0.01', '0.01_0.1', '>=0.1'])
    grams = top_grams['grams'].tolist()
    print('Successfully read the grams data and build the initial df!')

    del bigrams, unigrams, top_grams
    print(gc.collect())

    import functools
    args = zip(data['text'], data['idx'])
    func = functools.partial(build_grams, grams=grams)
    l_d = data.shape[0]
    with Pool() as p:
        ids = list(tqdm(p.imap(func, args, chunksize=2000), total=l_d))
    p.close()
    p.join()
    print('Successfully get the id list!')
    print(gc.collect())

    p_array = np.zeros((len(grams), 6))
    l_i = len(ids)
    price_v = price.values.astype(int)
    for i in tqdm(range(l_i)):
        if len(ids[i]) == 0:
            continue
        for j in ids[i]:
            m, n = j
            p_array[m] = p_array[m] + price_v[n]

    df = pd.DataFrame(p_array, columns=['<-0.1', '-0.1_0.01', '-0.01_0', '0_0.01', '0.01_0.1', '>=0.1'])
    df['grams'] = grams
    df.to_csv('top_grams2.csv')