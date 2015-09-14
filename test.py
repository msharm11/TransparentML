'''
Created on Sep 10, 2015

@author: Mustafa
'''

from classifiers import TransparentLogisticRegression
import numpy as np
import glob
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from scipy.sparse.construct import diags

def load_imdb(path, shuffle = True, random_state=42, vectorizer = CountVectorizer(min_df=5, max_df=1.0, binary=True)):
    
    print "Loading the imdb reviews data"
    
    train_neg_files = glob.glob(path+"\\train\\neg\\*.txt")
    train_pos_files = glob.glob(path+"\\train\\pos\\*.txt")
    
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
    
    test_neg_files = glob.glob(path+"\\test\\neg\\*.txt")
    test_pos_files = glob.glob(path+"\\test\\pos\\*.txt")
    
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
    ind_evi = (X_test[i]*coef_diags).toarray()[0]
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
    ind_evi = (X_test[i]*coef_diags).toarray()[0]
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
    ind_evi = (X_test[i]*coef_diags).toarray()[0]
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
    
    
    
    
    