#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import load
from imblearn.pipeline import Pipeline as imb_pipe

class Pipeline(imb_pipe):
    '''Pipeline of transforms and resamples with a final estimator.
    
    Extendes imblearn.pipeline.Pipeline class
    https://github.com/scikit-learn-contrib/imbalanced-learn/blob/12b2e0d/imblearn/pipeline.py#L27
    
    '''
    def __init__(self, steps, debug=False, seed=42):
        self.steps = steps
        self.debug = debug
        self.seed = seed
        self.print_debug("imblearn.pipeline.Pipeline init")
        super().__init__(steps=self.steps, verbose=self.debug)
        self.load()
        
    def print_debug(s):
        if self.debug:
            print(s)
        
    def load():
        self.print_debug("loading data from file")
        self.train_X, self.val_X, self.train_y, self.val_y = load.read_data(self.debug)
    
    def transform(transformer):
        self.transform = None
        if transformer:
            self.transform = transformer
            self.print_debug("applying transformer to train_X")
            self.train_X = transformer(self.train_X)
            self.print_debug("applying transformer to val_X")
            self.val_X = transformer(self.val_X)
        else:
            self.print_debug("no transformer provided; dropping review column")
            self.train_X = self.train_X.filter(
                ['ex_id', 'user_id', 'prod_id', 'rating', 'date'], axis='columns')
            self.val_X = self.val_X.filter(
                ['ex_id', 'user_id', 'prod_id', 'rating', 'date'], axis='columns')
    
    