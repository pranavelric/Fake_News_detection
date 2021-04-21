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
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import pandas as pd
import os
import gensim
import nltk 
from sklearn.linear_model import LogisticRegression




df=pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/CombiningAndExecution/final.csv")



import nltk
nltk.download('stopwords')



df=pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/CombiningAndExecution/final.csv")



import nltk
nltk.download('punkt')




for sentences in df['body']:
    print(sentences)



tokens = [nltk.word_tokenize(sentences) for sentences in df['body']]


model = gensim.models.Word2Vec(tokens, size=300, min_count=1, workers=4)
print("\n Training the word2vec model...\n")



model.save("word2vec.model")



model = gensim.models.Word2Vec.load("word2vec.model")
model.train(df.body, total_examples=len(df.body), epochs=4)



model.save("word2vecbody.model")
import numpy as np
np.load('word2vecbody.model.wv.vectors.npy')



df.columns



train = []

for sentences in df[df.columns[0:4]].values:
    train.extend(sentences)

tokens = [nltk.word_tokenize(str(sentences)) for sentences in train]




import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt



df=pd.read_csv("E:/Projects/ML_Project/FakeNewsDetection/Combining and Modeling/final.csv")
df.head()




cnt_pro = df['fakeness'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('fake', fontsize=12)
plt.xticks(rotation=90)
plt.show();




from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['body'] = df['body'].apply(cleanText)



train, test = train_test_split(df, test_size=0.3, random_state=42)
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['body']), tags=[r.fakeness]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['body']), tags=[r.fakeness]), axis=1)



train_tagged.values[3]



import multiprocessing
cores = multiprocessing.cpu_count()


model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])



for epoch in range(45):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors





# Using distributed bag of words
y_train_dbow, X_train_dbow = vec_for_learning(model_dbow, train_tagged)
y_test_dbow, X_test_dbow = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train_dbow, y_train_dbow)
y_pred_dbow = logreg.predict(X_test_dbow)
from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test_dbow, y_pred_dbow))
print('Testing F1 score: {}'.format(f1_score(y_test_dbow, y_pred_dbow, average='weighted')))



model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])



for epoch in range(45):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha


# Using distributed memory
y_train_dm, X_train_dm = vec_for_learning(model_dmm, train_tagged)
y_test_dm, X_test_dm = vec_for_learning(model_dmm, test_tagged)
logreg.fit(X_train_dm, y_train_dm)
y_pred_dm = logreg.predict(X_test_dm)
print('Testing accuracy %s' % accuracy_score(y_test_dm, y_pred_dm))
print('Testing F1 score: {}'.format(f1_score(y_test_dm, y_pred_dm, average='weighted')))



model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)




from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])




def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors




# Using both distributed memory and distributed bag of words
y_train, X_train = get_vectors(new_model, train_tagged)
y_test, X_test = get_vectors(new_model, test_tagged)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))




import scikitplot.plotters as skplt
def plot_cmat(y_test, y_pred):
    skplt.plot_confusion_matrix(y_test,y_pred)
    plt.show()
    
plot_cmat(y_test_dm, y_pred_dm)
print("confusion matrix of logistic regression of Doc2vec using distributed memory")
plot_cmat(y_test_dbow, y_pred_dbow)
print("confusion matrix of logistic regression of Doc2vec using distributed bag of words")
plot_cmat(y_test, y_pred)
print("confusion matrix of logistic regression of Doc2vec combining both models")



C = 1.0  
from sklearn import svm
svc = svm.SVC(kernel='linear', C=C).fit(X_train_dm, y_train_dm)
y_pred= svc.predict(X_test_dm)
print('Testing accuracy %s' % accuracy_score(y_test_dm, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test_dm, y_pred, average='weighted')))



plot_cmat(y_test_dm, y_pred)
print("confusion matrix of SVM of Doc2vec using distributed memory")

