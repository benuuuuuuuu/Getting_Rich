# Getting_Rich
HKU MSBA 7012 Social Media Project

Utilizing Twitter to Build Small-Cap Stock Trading Strategy
Class A, Group 9
 
This is a readme file for all the codes in the social media project for MSBA 7012.
 
Contributors: GUO Tianya, HUANG Wenjie, JIN Yu, LI Mengjin, 
LIU Qinlei, LU Meiqi, U Chi Lai, YAO Yao
 
Overview
All codes should at least run after preprocessing. Some codes could be run after preprocessing_total.py, but some need to run after 7012_data_handling_improved-v2-upload.ipynb if the code used stock price.
 
Data Link:
https://drive.google.com/drive/folders/1aLXluUmfcdoISOrsJjC0Pl5AqHAep65F?usp=sharing
 
Preprocess -> [Self_built_dict, Count Based Model, Sentiment Analysis, Topic Modelling, Word2Vector] -> Modelling

How to Read and Run the Codes:
Preprocess
File: proprecessing_total.py
Author: Yao Yao
Last update: 03/14/2021
 
---README---
This code will run with all of your CPUs!!! (changeable, see row 23).
You will see a progress monitor in the output box like this
    xx%| finished_iterations/total_iterations [already_run_time<estimated_remain_time, xx it/s]
input: train.sql & test.sql (in Google drive, import to mysql)
output: mysql table with posts updated to processed texts
*****************************************************************
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
*****************************************************************
Please change the followings before running this code:
1. connection details (see row 109 & 126)
2. [i, j] (see row 108)
3. if only want to run with n CPUs, change 'with Pool() as p' to 'with Pool(n) as p' (see row 111)
   to see how many CPUs your laptop has:
    import multiprocessing
    multiprocessing.cpu_count()
*****************************************************************
 
File: 7012_data_handling_improved-v2-upload.ipynb
Author: U Chi Lai
Last update: 03/14/2021
 
---README---
This code should follow by preprocessing_total.py
 
This code aims to handle the raw text and price data by matching them for further processes. The code imports the csv file of prices for stocks and twitter text data from SQL database. Please clear up the paths for input files and set up the appropriate sql database before running the code.

Self_built_dict
Author: JIN Yu (Kyle)
Last update: 03/14/2021
 
---README---
All this 8 .py files are for our self-built dictionaries and corresponding features
 
Step order:
1. preprocessing_total.py -> preprocessing
2. keep_1w_words.py -> keep the words in the common 10,000 words
3. build_grams.py -> get the top 2000 unigrams and top 2000 bigrams
4. build_each_ticker_date_class.py -> get D2D price change of each ticker-day (finance data)
5. match_ticker_date.py -> merge the finance data to twitter data
6. v2_sum_vote.py -> sum the vote of each twitter for the grams (probability density)
7. build_each_post_ML_data.py -> build 6 dicts and do feature engineering for each twitter
8. build_ML_feature.py -> sum ticker-date features
 
Tips:
1. File name may change.
2. Mysql connection details should be modified.
3. Pycharm is recommended	 for multiprocessing.


Sentiment Analysis
Author: Yao Yao
Last update: 03/14/2021

---README---
This code should be run after combining preprocessed twitter data and stock price, which means should be run after 7012_data_handling_improved-v2-upload.ipynb.

This code mainly handles sentiment analysis by using the other data in this file. There is also some EDA. If you only need the sentiment feature results, you can run and stop by the notice in that ipynb file. (You will see the notice in the file)



Topic Modelling
Author: Huang Wenjie
Last modified date: 2021/03/12
Environment : Python 3
 
Package:
1. Topic_Modeling_Gensim.py: gensim,nltk
2. Topic_Modeling_Spark.py: pyspark.context, pyspark.sql.session, pyspark.ml.feature, pyspark.ml.clustering, pyspark.sql.functions, pyspark.sql.types, pyspark.sql.function 
 
Conf for spark:
1. spark.driver.memory: 15g
2. spark.executor.memory: 10g
3. spark.memory.offHeap.enabled: True
4. spark.memory.offHeap.size: 2g
5. spark.driver.maxResultSize; 20g
6. spark.sql.autoBroadcastJoinThreshold: -1
 
Steps order:
1. Find the best number of topics.
2. Build the final LDA model.
3. Get the distribution and topic words.
 
Tips:
1. Gensim can handle the small dataset, however for the big dataset use the spark to build the model.
2. Pycharm is recommended.
 
Word2Vector
Author: Guo Tianya
Last modified date: 2021/03/12
 
To run the "w2v2.ipynb" file, please put the "closeeps.csv" in the same direction.
 
And the original data is based on our preprocessed dataset in mysql.
 
Also, please go to the following URL to download the "googlew2v.bin" file:
https://drive.google.com/file/d/1D2UthQ_Q-DKsk3BT80iHHJ_pSrK0os1A/view?usp=sharing
 
 
 
Modelling
Author: U Chi Lai
Last modified date: 2021/03/12
 
How to use:
1. Please put the social media folder in your google drive
2. Use google Colab to open it in order to make it work
3. Make sure you adjust the paths of the file in the codes
 
Codes:
1. model is the major code for building classification model and trading simulation models
2. fundamentals.ipynb retrieves the train/test company fundamentals 
3. match_text_closeprice_3daychange_lagprice.ipynb calculates the future 3 day price change for classification and the 1-day lag close price for inputs.
4. matching.ipynb matches features by ticker and date
5. ratios.ipynb retrieves the train/test financial ratios
6. rematch_train3dc.ipynb rematch the 3day price change for training data in order to confirm there is no error before building the classification model.
 
Tips:
1. Get the Colab Pro in order to efficiently run these codes with lightning-fast GPU acceleration :)

