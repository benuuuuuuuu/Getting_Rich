# -*- coding: utf-8 -*-
# author：Kyle time:8/3/2021
import gc
from multiprocessing import Pool, Manager
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm


def build_ML_data(arg, result):
    ticker, date = arg[0], arg[1]
    result[(ticker, date)] = result[(ticker, date)] + arg[2:].astype(int)


if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select ticker,date,g1,g2,g3,g4,g5,g6,n_word from pro_array_test"
    data = pd.read_sql(sqlcmd, conn)
    conn.close()
    print('Successfully read the sql data!')

    result = data[['ticker', 'date']].drop_duplicates(keep='first', subset=['ticker', 'date'], ignore_index=True)
    l_r = result.shape[0]
    print(l_r)
    result_m = Manager().dict()
    for i in tqdm(range(l_r)):
        result_m[(result.iloc[i, 0], result.iloc[i, 1])] = np.zeros((7,)).astype(int)
    del result

    import functools
    func = functools.partial(build_ML_data, result=result_m)
    args = list(data.values)
    del data
    print(gc.collect())

    l_d = len(args)
    with Pool() as p:
        list(tqdm(p.imap(func, args, chunksize=2000), total=l_d))
    p.close()
    p.join()
    del args
    print('Successfully build the data for ML!')

    df = pd.DataFrame(dict(result_m)).T
    df.reset_index(inplace=True)
    df.columns = ['ticker', 'date', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'n_word']
    print('Successfully transform to df!')

    conn = create_engine('mysql+pymysql://root:******@localhost/Twitter', encoding='utf8')
    pd.io.sql.to_sql(df, "ML_gram_feature_test", conn, if_exists='replace')
    print('All done!')
