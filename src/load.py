#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import pandas as pd
from pathlib import Path
    
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
        
    train = 'train.csv'
    val = 'dev.csv'

    try:
        if debug: 
            print("try:reading train.csv")
        train_df = pd.read_csv(train)
        if debug: 
            print("try:reading dev.csv")
        val_df = pd.read_csv(val)
    except:
        try:
            if debug:
                print("except-try: path update")
            data_folder = Path(os.path.dirname(__file__).replace('src', 'data'))
        except:
            if debug:
                print("except-except: path update")
            data_folder = Path(os.path.abspath('').replace('src', 'data'))
        finally:
            train = data_folder / train
            val = data_folder / val
            if debug: 
                print("except-finally: reading train.csv")
            train_df = pd.read_csv(train)
            train_df['date'] = pd.to_datetime(train_df['date'])
            if debug: 
                print("except-finally: reading dev.csv")
            val_df = pd.read_csv(val)
            val_df['date'] = pd.to_datetime(val_df['date'])
    
    if debug:
        print("data loaded - filtering DFs")
    
    X_col = ['ex_id', 'user_id', 'prod_id', 'rating', 'date', 'review']
    y_col = ['label']
    
    train_X = train_df.filter(X_col, axis='columns')
    train_y = train_df.filter(y_col, axis='columns')
    val_X = val_df.filter(X_col, axis='columns')
    val_y = val_df.filter(y_col, axis='columns')
    
    if debug:
        print("exiting read_data")
    
    return train_X, train_y, val_X, val_y