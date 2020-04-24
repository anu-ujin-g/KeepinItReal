#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

# references
# https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html#smote-adasyn
# https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb

#SMOTE
import imblearn.over_sampling


def SMOTE(df, seed=42, debug=False):
    '''Helper procedure to manage dataset class imbalance.
    @TODO
    
    Parameters
    ----------
    df    : dataframe
        Dataset to oversample with SMOTE
    
    seed  : integer
        Sets the random_state when sampling for reproducability; default=42
        
    debug : boolean to set debugging verbosity
    
    
    Returns
    -------
    Four pandas dataframes:
        train_X, val_X, train_y, val_y
    '''
    if debug: 
        print("inside SMOTE")
    
    sm = imblearn.over_sampling.SMOTENC(sampling_strategy='minority', 
                                        categorical_features=['review'],
                                        random_state=seed)
    
    if debug:
        print("oversampler instantiated")
    
    df_X, df_y = sm.fit_sample(df.drop('label', axis='columns'), df['label'])
    df = pd.concat([pd.DataFrame(df_X), pd.DataFrame(df_y)], axis='columns')
    
    if debug:
        print("oversampler fit_trained")
    
    return df
    
