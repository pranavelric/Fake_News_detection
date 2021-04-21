import pandas as pd
import numpy as np 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import re
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


dataset1=pd.read_csv(r"E:\Projects\ML_Project\FakeNewsDetection\fake.csv")
cols_of_interest=["author","title","text","language","site_url","type","published","uuid"]
fn=dataset1[cols_of_interest]
print(fn.shape)
fn.drop(fn[fn.language!="english"].index,inplace=True)
print(fn.shape)


print(fn.describe())



print(fn.type.unique())
print(fn.shape)
fn.head(100)



fn.dropna(subset=['text','title'],inplace=True)
print(fn.shape)
fn.head(100)





fn.info()


df=fn
df['fakeness'] = 1
df.head(60)



df=df.head(3)
df.head()




tdf = TfidfVectorizer(stop_words='english',ngram_range=(1,2) )
df=df.head(11000)
vectorizer = tdf.fit(df.text.values.astype('U'))
transformed_text = vectorizer.transform(df.text.values.astype('U'))
transformed_title = vectorizer.transform(df.title.values.astype('U'))
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

# print(tdf.get_feature_names())
# print("/n/n/n")
# print(tdf.vocabulary_)




def plotTfidfScores(scores,termNames, n_words = 18):
    fig = plt.figure(figsize = (14, 18))
    override = {'fontsize': 'large'}
    fig.add_subplot(221)   
    n_words = 75
    sb.set()
    sb.barplot(x = scores[:n_words], y = termNames[:n_words]);
    plt.title("TFIDF - Importance of Top {0} Terms".format(n_words));
    plt.xlabel("TFIDF Score");




plotTfidfScores(scores, termNames,  n_words = 18)


df.to_csv("E:/Projects/ML_Project/FakeNewsDetection/DataCleaning/kaggle_cleaned.csv")

