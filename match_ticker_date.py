# -*- coding: utf-8 -*-
# author：Kyle time:6/3/2021
import gc
from datetime import timedelta, datetime
from multiprocessing import Pool, Manager
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm


def price_con(i, data_m, price_m):
    ticker, date = data_m[i]
    arr = np.ones((6,))
    key = (ticker, date)
    arr0 = price_m.get(key, arr)
    if arr0.sum() == 1:
        return arr0
    key1 = (ticker, date + timedelta(days=-1))
    arr1 = price_m.get(key1, arr)
    if arr1.sum() == 1:
        return arr1
    key2 = (ticker, date + timedelta(days=-2))
    arr2 = price_m.get(key2, arr)
    if arr2.sum() == 1:
        return arr2
    return np.zeros((6,))


if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select ticker,created_time from clean2_test"
    sqlcmd2 = "select ticker,date,p1,p2,p3,p4,p5,p6 from pro_price"
    data = pd.read_sql(sqlcmd, conn)
    price = pd.read_sql(sqlcmd2, conn)
    conn.close()
    print('Successfully read the sql data!')

    data.columns = ['ticker', 'date']
    data.drop_duplicates(keep='first', inplace=True, subset=['ticker', 'date'], ignore_index=True)
    data['date'] = data['date'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))
    print('Successfully process the date!')

    import functools
    price_m = Manager().dict()
    data_m = Manager().list()
    for i in data.values:
        data_m.append(tuple(i))
    for i in price.values:
        price_m[(i[0], i[1])] = i[2:]
    func1 = functools.partial(price_con, data_m=data_m, price_m=price_m)
    l = data.shape[0]
    with Pool() as p:
        p_array = list(tqdm(p.imap(func1, range(l), chunksize=1000), total=l))
    p.close()
    p.join()
    print('Successfully get the p_array!')

    del price, data, price_m, data_m
    print(gc.collect())

    p_df = pd.DataFrame(np.vstack(p_array))
    p_df.columns = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
    conn = create_engine('mysql+pymysql://root:******@localhost/Twitter', encoding='utf8')
    pd.io.sql.to_sql(p_df, "price_concat_test", conn, if_exists='replace')
    print('All done!')
