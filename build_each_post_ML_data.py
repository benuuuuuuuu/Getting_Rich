# -*- coding: utf-8 -*-
# author：Kyle time:8/3/2021
import gc
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine


def build_data(doc, grams):
    uni = doc.split()
    n_word = len(uni)
    words = list()
    if n_word > 1:
        for i in range(n_word - 1):
            words.append(uni[i] + ' ' + uni[i + 1])
    words = words + uni
    l = len(grams)
    arr = np.zeros((l+1,))
    arr[l] = n_word
    for word in words:
        for i in range(l):
            if word in grams[i]:
                arr[i] = arr[i] + 1
    return arr


if __name__ == '__main__':
    df = pd.read_csv('top_grams2.csv', index_col=0)
    df.iloc[:, :-1] = df.iloc[:, :-1].astype(int)
    df.columns = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'gram']
    df['mean'] = df.iloc[:, :-1].values.mean(axis=1)
    df = df[[True if len(i) > 2 else False for i in df['gram']]].reset_index(drop=True)
    df2 = pd.DataFrame()
    df2['d1'] = (df['p6'] - df['p1']) / df['mean']
    df2['d2'] = (df['p5'] - df['p2']) / df['mean']
    gram1 = list(df[df2['d1'] > 0.07]['gram'])
    gram2 = [i for i in list(df[df2['d2'] > 0.1]['gram']) if i not in gram1]
    gram3 = [i for i in list(df[(df2['d2'] > 0.03) & (df2['d2'] < 0.1)]['gram']) if i not in gram1 + gram2]
    gram4 = list(df[df2['d1'] < -0.03]['gram'])
    gram5 = [i for i in list(df[df2['d2'] < 0]['gram']) if i not in gram4]
    gram6 = [i for i in list(df[(df2['d2'] < 0.03) & (df2['d2'] > 0)]['gram']) if i not in gram3 + gram4]
    grams = [gram1, gram2, gram3, gram4, gram5, gram6]
    del df, df2
    print(gc.collect())
    print('Successfully generate the gram dictionary!')
    print(len(gram1), len(gram2), len(gram3), len(gram4), len(gram5), len(gram6))

    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select text from clean2_test"
    text = pd.read_sql(sqlcmd, conn)
    conn.close()
    raw_text = text['text']
    print('Successfully read the text data!')

    import functools
    from multiprocessing import Pool
    from tqdm import tqdm
    func = functools.partial(build_data, grams=grams)
    with Pool() as p:
        g_array = list(tqdm(p.imap(func, raw_text, chunksize=2000), total=raw_text.shape[0]))
    p.close()
    p.join()
    del text
    print(len(g_array))
    print('Successfully get the array from raw text!')

    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select ticker,created_time from clean2_test"
    data = pd.read_sql(sqlcmd, conn)
    conn.close()
    print('Successfully read the sql data!')

    data.columns = ['ticker', 'date']
    data['g1'] = np.zeros((data.shape[0],))
    data['g2'] = np.zeros((data.shape[0],))
    data['g3'] = np.zeros((data.shape[0],))
    data['g4'] = np.zeros((data.shape[0],))
    data['g5'] = np.zeros((data.shape[0],))
    data['g6'] = np.zeros((data.shape[0],))
    data['n_word'] = np.zeros((data.shape[0],))
    data.iloc[:, 2:] = np.array(g_array)
    print('Successfully add the array to data!')
    conn = create_engine('mysql+pymysql://root:******@localhost/Twitter', encoding='utf8')
    pd.io.sql.to_sql(data, "pro_array_test", conn, if_exists='replace')
    print('All done!')