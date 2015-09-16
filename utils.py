'''
Created on Sep 16, 2015

@author: mbilgic
'''

import numpy as np
import glob
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import re

def load_imdb(path, shuffle=True, random_state=42, vectorizer=CountVectorizer(min_df=5, max_df=1.0, binary=True)):
    
    print "Loading the imdb reviews data"
    
    train_neg_files = glob.glob(path + "\\train\\neg\\*.txt")
    train_pos_files = glob.glob(path + "\\train\\pos\\*.txt")
    
    train_corpus = []
    
    y_train = []
    
    for tnf in train_neg_files:
        f = open(tnf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(0)
        f.close()
    
    for tpf in train_pos_files:
        f = open(tpf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(1)
        f.close()
    
    test_neg_files = glob.glob(path + "\\test\\neg\\*.txt")
    test_pos_files = glob.glob(path + "\\test\\pos\\*.txt")
    
    test_corpus = []
    
    y_test = []
    
    for tnf in test_neg_files:
        f = open(tnf, 'r')
        test_corpus.append(f.read())
        y_test.append(0)
        f.close()
    
    for tpf in test_pos_files:
        f = open(tpf, 'r')
        test_corpus.append(f.read())
        y_test.append(1)
        f.close()
        
    print "Data loaded."
    
    print "Extracting features from the training dataset using a sparse vectorizer"
    print "Feature extraction technique is %s." % vectorizer
    t0 = time()
    
    X_train = vectorizer.fit_transform(train_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_train.shape
    print
        
    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()
        
    X_test = vectorizer.transform(test_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_test.shape
    print
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))        
        
        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]
        
        
        indices = np.random.permutation(len(y_test))
        
        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
         
    return X_train, y_train, X_test, y_test, train_corpus_shuffled, test_corpus_shuffled

class ColoredDoc(object):
    def __init__(self, doc, feature_names, coefs):
        self.doc = doc
        self.feature_names = feature_names
        self.coefs = coefs
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ")        
        for token in tokens:
            vocab_tokens = self.token_pattern.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.feature_names.index(vocab_token)
                    if self.coefs[vocab_index] > 0:
                        html_rep = html_rep + "<font color=blue> " + token + " </font>"
                    elif self.coefs[vocab_index] < 0:
                        html_rep = html_rep + "<font color=red> " + token + " </font>"
                    else:
                        html_rep = html_rep + "<font color=gray> " + token + " </font>"
                except:
                    html_rep = html_rep + "<font color=gray> " + token + " </font>"
            else:
                html_rep = html_rep + "<font color=gray> " + token + " </font>"
        return html_rep
