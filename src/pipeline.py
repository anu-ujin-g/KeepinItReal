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
        
    def print_debug(self, s):
        if self.debug:
            print(s)
        
    def load(self):
        self.print_debug("loading data from file")
        self.train_X, self.val_X, self.train_y, self.val_y = load.read_data(self.debug)
    
    def transform(self, transformer):
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



'''    
Pipeline thoughts:
1. load train, val
2. Transform - variable function
    1. take in train, val df
    2. create transformer (off training only!!) 
    3. transform train, val df
    4. returns transformer function (we need for test data later), transformed_train, transformed_val df
3. Imbalance - variable (must happen after transforming to oversample)
    1. oversample
    2. downsample
    3. both(?)
4. load kfold class for training (kfold cv using train data only - overkill?) 
5. Train - variable function for each person
    1. declare model/classifier
    2. optimize hyper-parameters (grid_search/nelder-mead/etc)
    3. return optimized model
6. score (auROC/AP) & plot?
7. pickle best trained model + transformer (saves model so we can use it on test)
'''


