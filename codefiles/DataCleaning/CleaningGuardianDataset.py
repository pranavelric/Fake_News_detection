import zipfile
import os
import json
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import re
import seaborn as sb
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")




df = pd.read_json("E:/Projects/ML_Project/FakeNewsDetection/FetchingData/tempdata/articles/2018-01-01.json" , encoding='utf-8')



count = 0
for filename in os.listdir("E:/Projects/ML_Project/FakeNewsDetection/FetchingData/tempdata/articles"):
    count+=1
    if count>1:
        file_path = "E:/Projects/ML_Project/FakeNewsDetection/FetchingData/tempdata/articles/" + filename
        df_ = pd.read_json(file_path , encoding='utf-8')
        df = pd.concat(objs= [df,df_], axis=0,ignore_index=True)
        
print(df.shape)




lst = []
for i in range(df.shape[0]):
    lst.append(df.fields[i]["bodyText"])




lst_head = []
for i in range(df.shape[0]):
        lst_head.append(df.fields[i]["headline"])


df["headline"] = lst_head
df = df[df.bodyText != ""]


df.shape
df = df[(df.sectionName == 'US news') | (df.sectionName == 'Business') | (df.sectionName == 'Politics') | (df.sectionName == 'World news')]
df = df.reset_index(drop=True)
df.info()
df.describe
Counter(df.sectionName)

df.dropna(subset=['bodyText','headline'],inplace=True)
print(df.shape)



df.webPublicationDate.min() ,df.webPublicationDate.max()



df.to_csv(r"E:/Projects/ML_Project/FakeNewsDetection/DataCleaning/guardian_cleaned.csv")



df.info()


tdf = TfidfVectorizer(stop_words='english',ngram_range=(1,2) )
vectorizer = tdf.fit(df.bodyText)
transformed_text = vectorizer.transform(df.bodyText)
transformed_title = vectorizer.transform(df.headline)





def getTfidfTermScores(feature_names):
    term_corpus_dict = {}
    for term_ind, term in enumerate(feature_names):
        term_name = feature_names[term_ind]
        term_corpus_dict[term_name] = np.sum(transformed_title.T[term_ind].toarray())
        
    return term_corpus_dict


feature_names = tdf.get_feature_names()
term_corpus_dict = getTfidfTermScores(feature_names)





def getSortedTfidfScores(term_corpus_dict):

    sortedIndices = np.argsort( list(term_corpus_dict.values()))[::-1]
    termNames = np.array(list(term_corpus_dict.keys()))
    scores = np.array(list(term_corpus_dict.values()))
    termNames = termNames[sortedIndices]
    scores = scores[sortedIndices]
    return termNames, scores


termNames, scores = getSortedTfidfScores(term_corpus_dict)



def plotTfidfScores(scores,termNames, n_words = 18):
    fig = plt.figure(figsize = (14, 18))
    override = {'fontsize': 'large'}
    fig.add_subplot(221) 
    n_words = 75
    sb.set()
    sb.barplot(x = scores[:n_words], y = termNames[:n_words]);
    plt.title("TFIDF score ".format(n_words));
    plt.xlabel("TFIDF Score");



plotTfidfScores(scores, termNames,  n_words = 18)

