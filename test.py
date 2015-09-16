'''
Created on Sep 10, 2015

@author: Mustafa
'''

from classifiers import TransparentLogisticRegression
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from scipy.sparse.construct import diags

from utils import load_imdb

if __name__ == '__main__':
    # Load the data
    
    print "Loading the data"
    
    t0 = time()
    
    vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
    X_train, y_train, X_test, y_test, train_corpus, test_corpus = load_imdb("C:\\Users\\Mustafa\\Desktop\\aclImdb", shuffle=True, vectorizer=vect)
    feature_names = vect.get_feature_names()
    
    duration = time() - t0

    print
    print "Loading took %0.2fs." % duration
    print
    
    print "Fitting the classifier"
    
    t0 = time()
    clf = TransparentLogisticRegression(penalty='l1', C=0.1)
    clf.fit(X_train, y_train)
    
    duration = time() - t0

    print
    print "Fitting took %0.2fs." % duration
    print
    
    print "Predicting the evidences"
    
    t0 = time()
    neg_evi, pos_evi = clf.predict_evidences(X_test)
    
    duration = time() - t0

    print
    print "Predicting evidences took %0.2fs." % duration
    print
    
    print "Predicting the probs"
    
    t0 = time()
    probs = clf.predict_proba(X_test)
    
    duration = time() - t0

    print
    print "Predicting probs took %0.2fs." % duration
    print
    
    total_evi = neg_evi + pos_evi
    
    evi_sorted = np.argsort(total_evi)
    
    coef_diags = diags(clf.coef_[0], 0)
    
    # Most negative
    
    print
    print "Most negative"
    print
    i = evi_sorted[0]
    print total_evi[i], neg_evi[i], pos_evi[i], probs[i]
    print test_corpus[i]
    ind_evi = (X_test[i] * coef_diags).toarray()[0]
    ind_evi_sorted = np.argsort(ind_evi)
    print "Negative words"
    for j in ind_evi_sorted:
        if ind_evi[j] >= 0:
            break
        print feature_names[j], ind_evi[j]
    print "\nPositive words"
    for j in ind_evi_sorted[::-1]:
        if ind_evi[j] <= 0:
            break
        print feature_names[j], ind_evi[j]
    
    print
    print "Most positive"
    print
    i = evi_sorted[-1]
    print total_evi[i], neg_evi[i], pos_evi[i], probs[i]
    print test_corpus[i]
    ind_evi = (X_test[i] * coef_diags).toarray()[0]
    ind_evi_sorted = np.argsort(ind_evi)
    print "Negative words"
    for j in ind_evi_sorted:
        if ind_evi[j] >= 0:
            break
        print feature_names[j], ind_evi[j]
    print "\nPositive words"
    for j in ind_evi_sorted[::-1]:
        if ind_evi[j] <= 0:
            break
        print feature_names[j], ind_evi[j]
    
    

    
    print
    print "Least total absolute evidence"
    print
    total_abs_evi = abs(neg_evi) + pos_evi
    abs_evi_sorted = np.argsort(total_abs_evi)
    i = abs_evi_sorted[0]
    print total_evi[i], neg_evi[i], pos_evi[i], probs[i]
    print test_corpus[i]
    ind_evi = (X_test[i] * coef_diags).toarray()[0]
    ind_evi_sorted = np.argsort(ind_evi)
    print "Negative words"
    for j in ind_evi_sorted:
        if ind_evi[j] >= 0:
            break
        print feature_names[j], ind_evi[j]
    print "\nPositive words"
    for j in ind_evi_sorted[::-1]:
        if ind_evi[j] <= 0:
            break
        print feature_names[j], ind_evi[j]
    
    
    
    
    
