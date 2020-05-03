#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import load
import kir_models as km
import kir_transformers as kt

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
    
    transformers = ['cv', 'tfidf', 'w2v', 'bert']
    models = ['svm', 'nb', 'nn', 'lr']
    save_plots = True
    
    for t in transformers:
        t_ = eval('kt.'+t)(tx.review)
        tx_ = t_.transform(tx.review.copy())
        vx_ = t_.transform(vx.review.copy())
        for m in models:
            m_ = eval('km.'+m)().fit(tx_,ty)
            print(f'mean accuracy {t}|{m}: {m_.score(vx_,vy)}')
            if save_plots:
                km.metrics(m_, vx_, vy, f'{t}-{m}')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
