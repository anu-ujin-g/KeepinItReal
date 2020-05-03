#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Module returns models with best hyperparameters
'''
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_curve, auc



def svm(params=None):
    if not params:
        ## Using values from svm.py
        params = {
            'max_iter': 700,
            'C': 1,
            'gamma': 'auto',
            'probability': True
        }

    return SVC(**params)

def nb(params=None):
    if not params:
        ## No values specified in Feat_Eng_AG.ipynb; using BernoulliNB defaults
        params = {
            'alpha': 1.0, 
            'binarize': 0.0, 
            'class_prior': None, 
            'fit_prior': True
        }
    
    return BernoulliNB(**params)
        
def nn(params=None):
    if not params:
        ## Using values from nn.py
        params = {
            'activation': 'logistic', 
            'alpha': 0.1, 
            'hidden_layer_sizes': (255, 100, 50), 
            'learning_rate': 'constant', 
            'max_iter': 200, 
            'solver': 'sgd'
        }

    return MLPClassifier(**params)

def lr(params=None):
    if not params:
        ## No values specified ; using LogisticRegression defaults
        params = {
            'C': 1.0,
            'max_iter': 100,
            'penalty': 'l2',
            'solver': 'lbfgs',
        }

    return LogisticRegression(**params)
    
def metrics(clf, test_X, test_y, name):
    def plotAUC(preds, truth, score, name):
        if len(preds.shape) == 1:
            fpr, tpr, thresholds = roc_curve(truth, preds)
        else: 
            fpr, tpr, thresholds = roc_curve(truth, preds[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize = (8, 8))
        plt.plot(fpr, tpr, label=f'(AUC:{roc_auc:0.3f}, AP:{score:0.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('{name} - AUC')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.legend(loc="lower right")
        plt.savefig(f'{load.get_data_path()}_{name}.png')
    
    def printConfusionMatrix(preds, truth):
        print(pd.crosstab(truth.ravel(), preds, rownames=['True'],
                          colnames=['Predicted'], margins=True))
        
    y_truth, y_pred, y_prob = test_y, clf.predict(test_X), clf.predict_proba(test_X)
    score = average_precision_score(y_truth,y_prob[:, 1])
    
    printConfusionMatrix(y_truth, y_pred)
    plotAUC(y_truth, y_pred, score, name)


    
    