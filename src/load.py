#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import pandas as pd
from pathlib import Path

def get_data_path():
    '''Helper function to get data path within project.
    '''
    from pathlib import Path
    
    path = Path('.').resolve()
    path_string = path.absolute().as_posix()
    if 'src' in path_string:
        path = path.parent / 'data'
    elif 'data' in path_string:
        pass
    else:
        path = path / 'data'
    path_to_data = f'{path.absolute().as_posix()}/'
    return path_to_data

def read_test(test_data='test.csv', bert=False):
    path = get_data_path()
    test = path+test_data
    
    test_df = pd.read_csv(test)
    test_df['date'] = pd.to_datetime(test_df['date']).astype('int64')
    
    X_col = ['ex_id', 'user_id', 'prod_id', 'rating', 'date', 'review']
    test_X = test_df.filter(X_col, axis='columns')
    
    if bert:
        test_X = pd.read_pickle(path+'test_'+bert+'.pickle')
    
    return test_X
    

def read_data(train_data='train.csv', val_data='dev.csv', bert=False, debug=False):
    '''Helper procedure to load dataset.
    
    Parameters
    ----------
    train_data: string
        filename to load for training data
        
    val_data: string
        filename to load for validation data
        
    bert: string
        loads 'large'|'small' bert pickle data if set; default=False
    
    debug : boolean 
        sets debugging verbosity
    
    Returns
    -------
    Four pandas dataframes:
        train_X, val_X, train_y, val_y
    '''
    if debug:
        print("inside read_data")
            
    path = get_data_path()
    train = path+train_data
    val = path+val_data

    if debug: 
        print(f"reading {train}")
    train_df = pd.read_csv(train)
    if debug: 
        print(f"try:reading {val}")
    val_df = pd.read_csv(val)

    train_df['date'] = pd.to_datetime(train_df['date']).astype('int64')
    val_df['date'] = pd.to_datetime(val_df['date']).astype('int64')
    
    X_col = ['ex_id', 'user_id', 'prod_id', 'rating', 'date', 'review']
    y_col = ['label']
    
    if debug:
        print("spliting train,val")
        
    train_X = train_df.filter(X_col, axis='columns')
    val_X = val_df.filter(X_col, axis='columns')
    train_y = train_df.filter(y_col, axis='columns')
    val_y = val_df.filter(y_col, axis='columns')
    
    if bert:
        if debug: 
            print(f"reading {path}train_{bert}.pickle")
        train_X = pd.read_pickle(path+'train_'+bert+'.pickle')
        if debug: 
            print(f"try:reading {path}dev_{bert}.pickle")
        val_X = pd.read_pickle(path+'dev_'+bert+'.pickle')
    
    if debug:
        print("exiting read_data")
    
    return train_X, val_X, train_y, val_y