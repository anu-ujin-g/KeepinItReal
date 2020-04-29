#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# requires bert to be present: 
# python -m spacy download en_trf_bertbaseuncased_lg
import spacy
import load

def transform(df, save=False, name=''):
    from tqdm import tqdm
    tqdm.pandas()
    
    nlp = spacy.load('en_trf_bertbaseuncased_lg')
    bert = lambda x: nlp(x).vector
    
    
    df['review'] = df['review'].progress_apply(bert)
    
    if save:
        df.to_pickle(f'{load.get_data_path()}{name}_bert.pickle')

    return bert, df

## example how to run
# import load
# import bert
# df, val_X, train_y, val_y = load.read_data()
# df = df.sample(frac=0.001)
# df = df.reset_index(drop=True)
# nlp, df = bert.transform(df)


# # example how to run
# import load
# import bert
# train_X, val_X, train_y, val_y = load.read_data()
# nlp, df = bert.transform(train_X, True, 'train_X')
# nlp, df = bert.transform(val_X, True, 'val_X')