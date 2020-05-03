#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import load
import kit_models as km
import kit_transformers as kt

import warnings

def main():
    '''loads datasets into dataframes using load.py and prints head(2) for each
    '''
    print("reading data")  
    train_X, val_X, train_y, val_y = load.read_data()
    
    # downsample for testing
    print("downsampling to head(20)")
    tx = train_X.head(20).copy()
    vx = val_X.head(20).copy()
    ty = train_y.head(20).copy().values.ravel()
    vy = val_y.head(20).copy().values.ravel()
    
    transformers = ['kt.cv', 'kt.tfidf', 'kt.w2v', 'kt.bert']
    models = ['km.svm', 'km.nb', 'km.nn', 'km.lr']
    
    for t in transformers:
        t_ = eval(t)(tx.review)
        tx_ = t_.transform(tx.review.copy())
        vx_ = t_.transform(vx.review.copy())
        for m in models:
            print(f'mean accuracy {t}|{m}: {eval(m)().fit(tx_,ty).score(vx_,vy)}')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()