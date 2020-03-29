#!/usr/bin/env python
# coding: utf-8

# <font color='red'>QUABOUL Dorian - 3872944</font><br>
# <font color='red'>IKHLEF MOUHAMAD - 3870476</font>

# # Traitement Automatique de la Langue

# _Import de librairies_

# In[31]:


import codecs
import re
import string
import time
import unicodedata
import numpy as np
import sklearn.naive_bayes as bayes
import matplotlib.pyplot as plt
from sklearn import svm, linear_model as lin
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import FrenchStemmer


# _Liens utiles_

# ##### Chargement des données
# - http://www-connex.lip6.fr/~guigue/wikihomepage/pmwiki.php?n=Course.CourseTALTME3bow
# 
# ##### Nettoyage, analyse des données textuelles
# - https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4854971-nettoyez-et-normalisez-les-donnees
#     
#     

# ##### Méthodes utiles

# In[5]:


def read_file(fn):
    with codecs.open(fn,encoding="utf-8") as f:
        return f.read()


# In[6]:


def compteLignes(fname):
    count = 0
    with open(fname, 'r') as f:
        for line in f:
            count += 1
    return count


# In[7]:


def preprocess(s):
    ptc = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
    table = s.maketrans(ptc, ' '*len(ptc))
    #table = str.maketrans(string.punctuation + string.digits, ' '*(len(string.punctuation)+len(string.digits)))
    s = s.translate(table)

    return re.sub("\s"," ",re.sub("\s(?=\s)"," ",s))


# In[8]:


def stemmatisation(s) :
    stemmer = FrenchStemmer()
    words = []
    
    for w in s.split(" ") :
        if w != "" :
            words.append(stemmer.stem(w))
    
    new_s = ' '.join(words)

    
    return new_s


# In[9]:


def extract_labels_txt(corpus,n_lines):
    alltxts = []
    labels = np.ones(n_lines)
    
    for i in range(n_lines):
        line = corpus.readline()

        label = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",line)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",line)
        
        preprocess_txt = preprocess(txt)
        stema_txt = stemmatisation(preprocess_txt)

        if label.count('M') > 0:
            labels[i] = -1
        alltxts.append(stema_txt)

        
    return labels,alltxts


# In[10]:


def cross_validation(clf, X, y, n_splits=5):
    scores = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_svm.fit(X_train, y_train)
        scores.append(f1_score(y_test,clf_svm.predict(X_test)))
        
    return sum(scores)/n_splits


# In[11]:


def get_vocab_croissant(corpus, ngram, stop_words = None):
    vectorizer = CountVectorizer(stop_words = stop_words, ngram_range = (1,ngram))
    X = vectorizer.fit_transform(corpus).toarray()
    vocab = np.array(vectorizer.get_feature_names())
    return vocab[X.sum(0).argsort()]


# In[12]:


def get_len_vocab(corpus, ngram, stop_words = None):
    vectorizer = CountVectorizer(stop_words = stop_words, ngram_range = (1,ngram))
    vectorizer.fit_transform(corpus)
    return len(vectorizer.get_feature_names())


# In[13]:


def campagne_evaluation(corpus, labels, stop_words, ngram = 1, classifieur = "bayes"):
    if classifieur == "bayes":
        clf = bayes.MultinomialNB()
    elif classifieur == "svm":
        clf = svm.LinearSVC(max_iter = 2000)
    elif classifieur == "lin":
        clf = lin.LogisticRegression()
    else:
        raise ValueError("No classifier between, svm, bayes and lin")
        
    vocab = get_len_vocab(corpus, ngram)
    vocab_sw = get_len_vocab(corpus, ngram)
        
    ordinate = [[] for i in range(4)]
    abscissa = np.linspace(20, get_len_vocab(corpus, ngram), 20, dtype = int)
    abscissa_sw = np.linspace(20, get_len_vocab(corpus, ngram, stop_words), 20, dtype = int)
    
    # Sans stop words
    for step in abscissa:
        vectorizer = CountVectorizer(max_features=step, ngram_range=(1,ngram))
        tf_idf = TfidfVectorizer(max_features=step, ngram_range=(1,ngram))
        X_v = vectorizer.fit_transform(corpus)
        X_t = tf_idf.fit_transform(corpus)
        ordinate[0].append(cross_validation(clf, X_v, labels))
        ordinate[1].append(cross_validation(clf, X_t, labels))
             
    # Avec stop words
    for step in abscissa_sw:
        vectorizer = CountVectorizer(max_features=step, ngram_range=(1,ngram), stop_words=stop_words)
        tf_idf = TfidfVectorizer(max_features=step, ngram_range=(1,ngram), stop_words=stop_words)
        X_v = vectorizer.fit_transform(corpus)
        X_t = tf_idf.fit_transform(corpus)
        ordinate[2].append(cross_validation(clf, X_v, labels))
        ordinate[3].append(cross_validation(clf, X_t, labels))

    plt.show()
    plt.plot(abscissa, ordinate[0],label='TF')
    plt.plot(abscissa, ordinate[1],label='TFIDF')
    plt.plot(abscissa_sw, ordinate[2],label='stopword TF')
    plt.plot(abscissa_sw, ordinate[3],label='stopword TFIDF')
    plt.legend()
    plt.title("F1-score en fonction de nb mots conservés : "+str(ngram)+"-gram, "+classifieur)
    plt.xlabel("Nombre de mots conservés")
    plt.ylabel("F1-score")
    plt.show()


# In[14]:


def campagne_evaluation_2(corpus, labels, stop_words, ngram = 1, classifieur = "bayes"):
    if classifieur == "bayes":
        clf = bayes.MultinomialNB()
    elif classifieur == "svm":
        clf = svm.LinearSVC(max_iter = 2000)
    elif classifieur == "lin":
        clf = lin.LogisticRegression()
    else:
        raise ValueError("No classifier between, svm, bayes and lin")
        
    ordinate = [[] for i in range(4)]
    abscissa = np.linspace(20, get_len_vocab(corpus, ngram), 20, dtype = int)
    abscissa_sw = np.linspace(20, get_len_vocab(corpus, ngram, stop_words), 20, dtype = int)
    
    # Sans stop words
    for step in abscissa:
        vectorizer = CountVectorizer(max_features=step, ngram_range=(1,ngram))
        tf_idf = TfidfVectorizer(max_features=step, ngram_range=(1,ngram))
        X_v = vectorizer.fit_transform(corpus)
        X_t = tf_idf.fit_transform(corpus)
        ordinate[0].append(cross_validation(clf, X_v, labels))
        ordinate[1].append(cross_validation(clf, X_t, labels))
             
    # Avec stop words
    for step in abscissa_sw:
        vectorizer = CountVectorizer(max_features=step, ngram_range=(1,ngram), stop_words=stop_words)
        tf_idf = TfidfVectorizer(max_features=step, ngram_range=(1,ngram), stop_words=stop_words)
        X_v = vectorizer.fit_transform(corpus)
        X_t = tf_idf.fit_transform(corpus)
        ordinate[2].append(cross_validation(clf, X_v, labels))
        ordinate[3].append(cross_validation(clf, X_t, labels))

    plt.show()
    plt.plot(abscissa, ordinate[0],label='TF')
    plt.plot(abscissa, ordinate[1],label='TFIDF')
    plt.plot(abscissa_sw, ordinate[2],label='stopword TF')
    plt.plot(abscissa_sw, ordinate[3],label='stopword TFIDF')
    plt.legend()
    plt.title("F1-score en fonction de nb mots conservés : "+str(ngram)+"-gram, "+classifieur)
    plt.xlabel("Nombre de mots conservés")
    plt.ylabel("F1-score")
    plt.show()


# ### Tâche 1 : détection d'auteur, Chirac/Miterrand

# In[15]:


fname_train = "data/corpus.tache1.learn.utf8"
fname_test = "data/corpus.tache1.test.utf8"
file_train = codecs.open(fname_train,"r",encoding="utf-8")
file_test = codecs.open(fname_test,"r",encoding="utf-8")


# In[16]:


n_train = compteLignes(fname_train)
n_test = compteLignes(fname_test)
print("Nombre de lignes pour le corpus de train = %d"%n_train)
print("Nombre de lignes pour le corpus de test = %d"%n_test)


# In[17]:


labels_train, corpus_train = extract_labels_txt(file_train,n_train)
labels_test, corpus_test = extract_labels_txt(file_test,n_test)

vocab = get_vocab_croissant(corpus_train, 1)
print(vocab[:-2])

