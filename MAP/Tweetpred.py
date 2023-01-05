import warnings
import logging
warnings.filterwarnings('ignore')
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import word2vec, FastText
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import _stop_words


#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import sys
import sys;
# Visualization
import seaborn as sns
# Data processing
import pandas as pd
import numpy as np
# Model performance evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_log_error

from sklearn.decomposition import PCA, NMF,FastICA,TruncatedSVD,IncrementalPCA
from sklearn.cluster import KMeans
#import pandasgui
#from utils.utility_fct import get_table_sql,cross_product
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import unidecode


def preprocess(df):
    clean_txt = []
    for w in range(len(df.text)):
        text = df['text'][w].lower()
        #Due to the merge between texts and hashtags
        text = text.replace("["," ")
        text = text.replace("]","")
        text = text.replace("'","")
        text = text.replace(",","")
        text = unidecode.unidecode(text )
        clean_txt.append(text)
    df['clean'] = clean_txt
    return df

if __name__ == '__main__':
    data = pd.read_csv('train.csv', header = 0)
    nltk.download('stopwords')
    data["text"] = data["text"] + data["hashtags"].astype(str)
    df1 = preprocess(data)

    #tf-idf vectorization of text
    corpus = df1['clean'].values
    vectorizer = TfidfVectorizer(max_features=100, stop_words=stopwords.words('french'))
    X_ = vectorizer.fit_transform(corpus)
    temp = pd.DataFrame(X_.todense())
    #kmeans
    scl = StandardScaler()
    dfnorm = scl.fit_transform(temp)
    true_k = 68
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(dfnorm)
    preds = model.predict(dfnorm)
    other_predictors = ['favorites_count', 'followers_count', 'statuses_count', 'friends_count']  # 'favorites_count'
    tab = pd.concat([pd.DataFrame(preds), df1[other_predictors]], axis=1)
    Scaler = StandardScaler()
    ScalerFit = Scaler.fit(tab)
    tab = ScalerFit.transform(tab)


    # Split the data into training and testing set
    # X_train, X_test, y_train, y_test = train_test_split(tab, data['retweets_count'], test_size=0.1, random_state=428)
    rf = RandomForestRegressor()  # (**param)


    # Fitting the model on Training Data
    rf_b = rf.fit(tab, data['retweets_count'])  # rf.fit(X_train, y_train)


    # Measuring Goodness of fit in Training data
    coef_determ = r2_score(data['retweets_count'], rf_b.predict(tab))
    print('Random Forest Regressor R2 Value over the train set', coef_determ)


    NewSampleData = pd.read_csv('evaluation.csv', header=0)  # index_col=0
    EvalData = NewSampleData.copy()
    EvalData["text"] = EvalData["text"] + EvalData["hashtags"].astype(str)
    dframe = preprocess(EvalData)
    X_Eval = vectorizer.transform(dframe['clean'].values)
    temp2 = pd.DataFrame(X_Eval.todense())
    dfnorm_Eval = scl.fit_transform(temp2)
    preds2 = model.predict(dfnorm_Eval)
    tab2 = pd.concat([pd.DataFrame(preds2), EvalData[other_predictors]], axis=1)
    tab2 = ScalerFit.transform(tab2)


    # Generating Predictions
    Predict = rf_b.predict(tab2)
    PredictResult = pd.DataFrame(Predict, columns=['retweets_count'])
    PredictResult['TweetID'] = EvalData['TweetID']
    PredictResult.set_index('TweetID').reset_index(drop=False)
    PredictResult.to_csv('ModelClusterRForest.csv', index=False)

