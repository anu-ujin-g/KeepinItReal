#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Module returns models with best hyperparameters
'''
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


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
    
    
    
    