# -*- coding: utf-8 -*-
# author：Kyle time:2/3/2021


def keep_common(text, common):
    words = text.split()
    text = [word for word in words if word in common]
    text = ' '.join(text)
    return text


if __name__ == "__main__":
    common_words = list()
    with open('google-10000-english.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            common_words.append(line.strip())
    print('Successfully read the common words!')
    import pymysql
    import pandas as pd
    conn = pymysql.connect(host='localhost', user='root', password='******', database='Twitter')
    sqlcmd = "select ticker,created_time,text from clean_test"
    data = pd.read_sql(sqlcmd, conn)
    conn.close()
    print('Successfully read the data!')
    raw_text = data['text']
    import functools
    from multiprocessing import Pool
    from sqlalchemy import create_engine
    from tqdm import tqdm
    func = functools.partial(keep_common, common=common_words)
    with Pool() as p:
        pro_text = list(tqdm(p.imap(func, raw_text, chunksize=2000), total=raw_text.shape[0]))
    p.close()
    p.join()
    print('Successfully process the data!')
    data['text'] = pro_text
    conn = create_engine('mysql+pymysql://root:******@localhost/Twitter', encoding='utf8')
    pd.io.sql.to_sql(data, "clean2_test", conn, if_exists='replace')
    print('All done!')