# -*- coding: utf-8 -*-
# author：Kyle time:6/3/2021
import gc
from multiprocessing import Pool
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from tqdm import tqdm


def price_change(ticker, df):
    df = df[df['ticker'] == ticker].copy()
    next_price = list()
    for day in df['date']:
        d = day
        n = 0
        while n < 3:
            try:
                d = d + timedelta(days=1)
                next_price.append(df[df['date'] == d]['price'].values[0])
                break
            except:
                n += 1
                pass
        if n == 3:
            next_price.append(df[df['date'] == day]['price'].values[0])
    df['next_p'] = next_price
    df['d2d'] = (df['next_p'] - df['price']) / df['price']
    df['p1'] = (df['d2d'] < -0.1).astype(int)
    df['p2'] = ((-0.1 <= df['d2d']) & (df['d2d'] < -0.01)).astype(int)
    df['p3'] = ((-0.01 <= df['d2d']) & (df['d2d'] < 0)).astype(int)
    df['p4'] = ((0 <= df['d2d']) & (df['d2d'] < 0.01)).astype(int)
    df['p5'] = ((0.01 <= df['d2d']) & (df['d2d'] < 0.1)).astype(int)
    df['p6'] = (0.1 <= df['d2d']).astype(int)
    return df


if __name__ == '__main__':
    price = pd.read_csv('price.csv')
    print('Successfully read the price data!')
    price = price[['tic', 'datadate', 'prccd']]
    price.columns = ['ticker', 'date', 'price']
    price.reset_index(drop=True, inplace=True)
    print(gc.collect())
    price['ticker'] = price['ticker'].apply(lambda x: '$' + x)
    price['date'] = price['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
    print(gc.collect())
    print('Successfully process the ticker name and date!')
    import functools
    func = functools.partial(price_change, df=price)
    tickers = price['ticker'].unique()
    with Pool() as p:
        pro_price = list(tqdm(p.imap(func, tickers, chunksize=10), total=tickers.shape[0]))
    p.close()
    p.join()
    del price
    print(gc.collect())
    print('Successfully process the price!')
    pro_price = pd.concat(pro_price, ignore_index=True)
    gc.collect()
    print('Successfully concat the dataframe!')
    conn = create_engine('mysql+pymysql://root:******@localhost/Twitter', encoding='utf8')
    pd.io.sql.to_sql(pro_price, "pro_price", conn, if_exists='replace')
    print('Congratulations! All done!')