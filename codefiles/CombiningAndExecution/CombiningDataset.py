import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, accuracy_score 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import sys
import csv



df_guard = pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/DataCleaning/guardian_cleaned.csv")



df_guard["fakeness"] = 0
df_guard.columns



df_kaggle = pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/DataCleaning/kaggle_cleaned.csv")

df_kaggle.columns



df_nyt=pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/DataCleaning/NYT_cleaned.csv")


df_nyt.columns

df_nyt["fakeness"]=0
df_nyt.drop(['Unnamed: 0','index','abstract','byline.original', 'byline.person','byline.organization','document_type','keywords', 'lead_paragraph', 'multimedia', 'news_desk', 'print_page',
       'print_section','type_of_material','section_name', 'snippet', 'source',
       'subsection_name', 'type_of_material','word_count',
       'Unnamed: 23', 'Unnamed: 24','uri','web_url'] , inplace=True , axis=1)

df_nyt.columns




df_guard.drop(['Unnamed: 0', 'apiUrl', 'fields', 
        'isHosted', 'sectionId', 'sectionName', 'type',
         'webTitle', 'webUrl','pillarId','pillarName'],inplace=True,axis=1)



df_guard.columns

df_kaggle.drop([ 'Unnamed: 0',  'language', 'site_url' ,'type','author'],inplace=True,axis=1)
df_kaggle.columns



df_guard = df_guard.rename(columns={'bodyText' : 'body','webPublicationDate':'published'})
df_kaggle = df_kaggle.rename(columns={'text':'body','title':'headline','uuid':'id'})
df_nyt=df_nyt.rename(columns={'_id':'id','pub_date':'published'})
df_kaggle.columns,df_guard.columns,df_nyt.columns


print("The number of genuine articles in kaggle dataset are "+str(df_kaggle.shape[0]))


df_kaggle.head()


print("The number of articles in Gaurdian dataset are "+str(df_guard.shape[0]))
print("The number of articles in NYT dataset are "+str(df_nyt.shape[0]))
print("The number of articles in Kaggle dataset are "+str(df_kaggle.shape[0]))



df_guard.head()





df_nyt.head()




df = df_kaggle.append(df_guard, ignore_index=True)
df=df.append(df_nyt,ignore_index=True)



df.dropna(inplace=True)
print(df.shape)
df.info()



df.head()



df.to_csv("E:/Projects/ML_Project/FakeNewsDetection/CombiningAndExecution/final.csv")
df=pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/CombiningAndExecution/final.csv")




df=pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/CombiningAndExecution/final.csv")


df.shape


df = df.head(22200)
print(df.shape)
tdf = TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_df=0.85, min_df=0.01 )
X_body=tdf.fit_transform(df.body.values.astype('U'))
X_headline=tdf.fit_transform(df.headline.values.astype('U'))
Y=df.fakeness
