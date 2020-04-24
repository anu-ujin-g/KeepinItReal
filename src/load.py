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

def read_data(debug=False):
    '''Helper procedure to load dataset.
    
    Parameters
    ----------
    balance : boolean
        Specifies whether to apply SMOTE (from balance.py) to the data
    
    debug : boolean to set debugging verbosity
    
    Returns
    -------
    Four pandas dataframes:
        train_X, val_X, train_y, val_y
    '''
    if debug:
        print("inside read_data")
            
    path = get_data_path()
    train = path+'train.csv'
    val = path+'dev.csv'

    if debug: 
        print(f"reading {train}")
    train_df = pd.read_csv(train)
    if debug: 
        print(f"try:reading {val}")
    val_df = pd.read_csv(val)

    train_df['date'] = pd.to_datetime(train_df['date'])
    val_df['date'] = pd.to_datetime(val_df['date'])
    
    X_col = ['ex_id', 'user_id', 'prod_id', 'rating', 'date', 'review']
    y_col = ['label']
    
    if debug:
        print("spliting train,val")
        
    train_X = train_df.filter(X_col, axis='columns')
    val_X = val_df.filter(X_col, axis='columns')
    train_y = train_df.filter(y_col, axis='columns')
    val_y = val_df.filter(y_col, axis='columns')
    
    if debug:
        print("exiting read_data")
    
    return train_X, val_X, train_y, val_y