'''
Created on Sep 10, 2015

@author: Mustafa
'''

from sklearn.linear_model import LogisticRegression
from scipy.sparse import diags, issparse
import numpy as np

class TransparentClassifier(object):
    
    def predict_evidences(self, X):
        pass

class TransparentLogisticRegression(TransparentClassifier, LogisticRegression):
    '''
    Transparent logistic regression
    '''

    def predict_evidences(self, X):
        coef_diags = diags(self.coef_[0],0)
        dm = X*coef_diags
        if issparse(dm):
            pos_evi = dm.multiply(dm>0).sum(1).A1
            neg_evi = dm.multiply(dm<0).sum(1).A1
        else:
            pos_evi = np.multiply(dm>0).sum(1).A1
            neg_evi = np.multiply(dm<0).sum(1).A1        
        return neg_evi, pos_evi
        
        