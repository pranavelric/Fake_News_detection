import pandas as pd
from collections import Counter
import re
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")



df = pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/FetchingData/NYT_Articles.csv")


df.info()


df = df.reset_index()

df.headline[29]



df.head()


df.to_csv("E:/Projects/ML_Project/FakeNewsDetection/DataCleaning/NYT_cleaned.csv")
df=pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/DataCleaning/NYT_cleaned.csv")



tdf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
vectorizer = tdf.fit(df.body.values.astype('U'))
transformed_text = vectorizer.transform(df.body.values.astype('U'))
transformed_title = vectorizer.transform(df['headline'].values.astype('U'))




def getTFidfTermScores(feature_names):
    term_corpus_dict = {}
    for term_ind, term in enumerate(feature_names):
        term_name = feature_names[term_ind]
        term_corpus_dict[term_name] = np.sum(transformed_title.T[term_ind].toarray())
        
    return term_corpus_dict



feature_names = tdf.get_feature_names()




term_corpus_dict = getTFidfTermScores(feature_names)




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
    plt.title(" Top tfidf score words".format(n_words));
    plt.xlabel("TFIDF Score");




plotTfidfScores(scores, termNames, n_words = 18)

